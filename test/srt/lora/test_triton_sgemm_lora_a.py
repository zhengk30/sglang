import unittest

import torch
import triton

from sglang.srt.lora.triton_ops.sgemm_lora_a_chunked import (
    _sgemm_lora_a_kernel_chunked,
)
from sglang.srt.lora.triton_ops.sgemm_lora_a import (
    _sgemm_lora_a_kernel as _sgemm_lora_a_kernel_non_chunked,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase


class TestTritonSgemmLoraA(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16

    @staticmethod
    def reorder_and_prepare_chunks(weight_indices, seg_lens, chunk_size: int):
        # Create a weight index for each row by repeating weight_indices according to seg_lens
        row_weight_indices = torch.repeat_interleave(weight_indices, seg_lens)

        # Sort rows by weight index (stable sort keeps relative order within each weight)
        index_map = torch.argsort(row_weight_indices, stable=True)

        # Get reordered weights to find group boundaries
        weights_reordered = row_weight_indices[index_map]

        # Get unique weights and their counts
        unique_weights, counts = torch.unique_consecutive(
            weights_reordered, return_counts=True
        )

        # Build chunk arrays
        chunk_to_weight = []
        cu_chunk_lens = [0]

        cumulative_pos = 0
        for weight_idx, group_len in zip(unique_weights, counts):
            group_len = group_len.item()
            num_chunks = (group_len + chunk_size - 1) // chunk_size

            chunk_to_weight.extend([weight_idx.item()] * num_chunks)

            # Add boundaries for each chunk
            for i in range(1, num_chunks):
                cu_chunk_lens.append(cumulative_pos + i * chunk_size)
            cu_chunk_lens.append(cumulative_pos + group_len)

            cumulative_pos += group_len

        chunk_to_weight = torch.tensor(
            chunk_to_weight, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        cu_chunk_lens = torch.tensor(
            cu_chunk_lens, dtype=torch.int32, pin_memory=True, device="cpu"
        )

        return index_map, chunk_to_weight, cu_chunk_lens

    def create_batch_info(self, bs):
        """
        Create batch info similar to how it is handled in production.
        This creates a LoRABatchInfo following the production patterns.
        """
        # Create basic batch info structure - use fixed sequence lengths for simplicity
        seg_lens = torch.randint(8, 32, (bs,), dtype=torch.int32, device="cpu")
        total_tokens = seg_lens.sum().item()

        # Create proper seg_indptr for non-chunked kernel
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
        seg_indptr[1:] = torch.cumsum(seg_lens.to(self.device), dim=0)

        # Weight indices - all sequences use the same adapter (index 0)
        weight_indices = torch.zeros((bs,), dtype=torch.int32, device="cpu")

        # LoRA ranks - single adapter with rank 64
        lora_ranks = torch.zeros((1,), dtype=torch.int64, device=self.device)
        lora_ranks[0] = 64  # rank of the single adapter

        # Scalings - single adapter with default scaling
        scalings = torch.ones((1,), dtype=torch.float, device=self.device)

        # Use the production reorder_and_prepare_chunks function to create proper chunking
        index_map, chunk_to_weight, cu_chunk_lens = (
            TestTritonSgemmLoraA.reorder_and_prepare_chunks(
                weight_indices, seg_lens, chunk_size=16
            )
        )

        # Move tensors to device
        index_map = index_map.to(self.device)
        chunk_to_weight = chunk_to_weight.to(self.device)
        cu_chunk_lens = cu_chunk_lens.to(self.device)

        # Number of chunks
        num_chunks = len(chunk_to_weight)

        # Get max sequence length
        max_len = seg_lens.max().item()

        return LoRABatchInfo(
            bs=bs,
            is_decode=False,  # extend mode for testing
            seg_lens=seg_lens.to(self.device),
            seg_indptr=seg_indptr,
            max_len=max_len,
            weight_indices=weight_indices.to(self.device),
            lora_ranks=lora_ranks,
            scalings=scalings,
            index_map=index_map,
            chunk_to_weight=chunk_to_weight,
            cu_chunk_lens=cu_chunk_lens,
            num_chunks=num_chunks,
        )

    def test_chunked_kernel_basic(self):
        """Test the _sgemm_lora_a_kernel function directly with simple inputs."""
        # Simple test case: single sequence, single LoRA adapter
        bs = 32
        input_dim = 4096
        rank = 64
        stack_num = 3

        # Create batch info for a single sequence
        batch_info = self.create_batch_info(bs)

        # Create test tensors
        total_len = len(batch_info.index_map)
        x = torch.randn((total_len, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn(
            (8, rank * stack_num, input_dim), device=self.device, dtype=self.dtype
        )
        output = torch.zeros(
            (total_len, rank * stack_num), device=self.device, dtype=x.dtype
        )

        # Block sizes used in the kernel
        BLOCK_S = 16
        BLOCK_N = 16
        BLOCK_K = 256

        # Launch the kernel with a simple grid
        grid = (
            triton.cdiv(rank * stack_num, BLOCK_N),  # N dimension
            batch_info.num_chunks,
        )

        _sgemm_lora_a_kernel_chunked[grid](
            x,
            weights,
            output,
            rank * stack_num,  # N
            input_dim,  # K
            stack_num,
            x.stride(0),
            x.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            batch_info.cu_chunk_lens,
            batch_info.index_map,
            batch_info.chunk_to_weight,
            batch_info.lora_ranks,
            batch_info.num_chunks,
            BLOCK_S,
            BLOCK_N,
            BLOCK_K,
        )

        # Compute reference result using PyTorch
        expected = torch.mm(x, weights[0].T)

        # Check that kernel output matches reference
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)

    def test_performance_comparison(self):
        """Compare performance between chunked and non-chunked kernels with same input data."""
        import time

        # Test parameters
        bs = 64  # Reduce batch size to avoid memory issues
        input_dim = 4096
        rank = 64
        stack_num = 3
        num_warmup = 1
        num_iterations = 50

        # Create batch info and input data (same for both kernels)
        batch_info = self.create_batch_info(bs)
        total_tokens = batch_info.seg_indptr[
            -1
        ].item()  # Total number of tokens across all sequences
        x = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        weights = torch.randn(
            (8, rank * stack_num, input_dim), device=self.device, dtype=self.dtype
        )

        # Common parameters
        N = rank * stack_num
        K = input_dim
        BLOCK_S = 16
        BLOCK_N = 16
        BLOCK_K = 256

        # Test chunked kernel
        output_chunked = torch.zeros(
            (total_tokens, N), device=self.device, dtype=x.dtype
        )
        grid_chunked = (triton.cdiv(N, BLOCK_N), batch_info.num_chunks)

        # Warmup chunked
        for _ in range(num_warmup):
            _sgemm_lora_a_kernel_chunked[grid_chunked](
                x,
                weights,
                output_chunked,
                N,
                K,
                stack_num,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                output_chunked.stride(0),
                output_chunked.stride(1),
                batch_info.cu_chunk_lens,
                batch_info.index_map,
                batch_info.chunk_to_weight,
                batch_info.lora_ranks,
                batch_info.num_chunks,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
            )
        torch.cuda.synchronize()

        grid_non_chunked = (
            triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
            batch_info.bs,
        )
        # Benchmark chunked
        start_time = time.time()
        for _ in range(num_iterations):
            _sgemm_lora_a_kernel_chunked[grid_non_chunked](
                x,
                weights,
                output_chunked,
                N,
                K,
                stack_num,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                output_chunked.stride(0),
                output_chunked.stride(1),
                batch_info.cu_chunk_lens,
                batch_info.index_map,
                batch_info.chunk_to_weight,
                batch_info.lora_ranks,
                batch_info.num_chunks,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
            )
        torch.cuda.synchronize()
        chunked_time = time.time() - start_time

        # Test non-chunked kernel
        output_non_chunked = torch.zeros(
            (total_tokens, N), device=self.device, dtype=x.dtype
        )
        grid_non_chunked = (
            triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
            batch_info.bs,
        )

        # Warmup non-chunked
        for _ in range(num_warmup):
            _sgemm_lora_a_kernel_non_chunked[grid_non_chunked](
                x,
                weights,
                output_non_chunked,
                N,
                K,
                stack_num,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                output_non_chunked.stride(0),
                output_non_chunked.stride(1),
                batch_info.seg_lens,
                batch_info.seg_indptr,
                batch_info.weight_indices,
                batch_info.lora_ranks,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
            )
        torch.cuda.synchronize()

        # Benchmark non-chunked
        start_time = time.time()
        for _ in range(num_iterations):
            _sgemm_lora_a_kernel_non_chunked[grid_non_chunked](
                x,
                weights,
                output_non_chunked,
                N,
                K,
                stack_num,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                output_non_chunked.stride(0),
                output_non_chunked.stride(1),
                batch_info.seg_lens,
                batch_info.seg_indptr,
                batch_info.weight_indices,
                batch_info.lora_ranks,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
            )
        torch.cuda.synchronize()
        non_chunked_time = time.time() - start_time

        # Print debug info
        print(f"Debug info:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Index map length: {len(batch_info.index_map)}")
        print(f"  Num chunks: {batch_info.num_chunks}")
        print(f"  Grid chunked: {grid_chunked}")
        print(f"  Grid non-chunked: {grid_non_chunked}")

        # Verify outputs are close (same computation, different organization)
        # Use a more relaxed tolerance since the kernels may have slightly different numerical behavior
        torch.testing.assert_close(
            output_chunked, output_non_chunked, atol=1e-2, rtol=1e-2
        )

        # Print performance results
        avg_chunked_time = chunked_time / num_iterations * 1000  # ms
        avg_non_chunked_time = non_chunked_time / num_iterations * 1000  # ms
        speedup = avg_non_chunked_time / avg_chunked_time

        print(f"\n=== Performance Comparison ===")
        print(
            f"Batch size: {bs}, Input dim: {input_dim}, Rank: {rank}, Stack num: {stack_num}"
        )
        print(f"Total tokens: {total_tokens}")
        print(f"Chunked kernel:     {avg_chunked_time:.3f} ms")
        print(f"Non-chunked kernel: {avg_non_chunked_time:.3f} ms")
        print(
            f"Speedup: {speedup:.2f}x {'(chunked faster)' if speedup > 1 else '(non-chunked faster)'}"
        )
        print(f"==============================\n")


if __name__ == "__main__":
    unittest.main()
