import unittest

import torch

from sglang.srt.lora.triton_ops import (
    chunked_sgmv_lora_expand_forward,
    chunked_sgmv_lora_shrink_forward,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_utils import CustomTestCase


class TestChunkedSgmvBackend(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
    
    def _test_chunked_sgmv_ops(
        self,
        bs=4,
        seq_len=32,
        input_dim=2048,
        rank=64,
        q_dim=2048,
        kv_dim=256,
        num_slices=3,
        segment_size=None,
        num_adapters=1,
        adapter_indices=None,
        lora_ranks=None,
        scalings=None,
        with_base_output=False,
        atol=1e-2,
        rtol=1e-2,
    ):
        """
        Unified test function that exercises both shrink and expand operations
        and compares them with torch reference implementations.
        
        Args:
            bs: Batch size
            seq_len: Sequence length
            input_dim: Input dimension for shrink operation
            rank: LoRA rank (used if lora_ranks is None)
            q_dim: Query dimension
            kv_dim: Key/Value dimension
            num_slices: Number of slices (e.g., 3 for Q/K/V, 2 for gate/up)
            segment_size: Segment size for chunking (None means no chunking within sequences)
            num_adapters: Number of LoRA adapters
            adapter_indices: Which adapter each sequence uses (None means all use adapter 0)
            lora_ranks: Custom ranks for each adapter (None means all use 'rank')
            scalings: Custom scalings for each adapter (None means all use 1.0)
            with_base_output: Whether to test with base output accumulation
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
        """
        total_tokens = bs * seq_len
        total_output_dim = q_dim + 2 * kv_dim if num_slices == 3 else q_dim
        
        # Create batch info
        if segment_size is None:
            # Simple case: one segment per sequence
            num_segments = bs
            seg_lens = torch.full((bs,), seq_len, dtype=torch.int32, device=self.device)
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
            
            if adapter_indices is None:
                weight_indices = torch.zeros((bs,), dtype=torch.int32, device=self.device)
            else:
                weight_indices = torch.tensor(adapter_indices, dtype=torch.int32, device=self.device)
        else:
            # Chunked case: multiple segments per sequence
            num_segs_per_seq = (seq_len + segment_size - 1) // segment_size
            num_segments = bs * num_segs_per_seq
            
            seg_lens_list = []
            for _ in range(bs):
                remaining = seq_len
                for _ in range(num_segs_per_seq):
                    seg_len = min(segment_size, remaining)
                    seg_lens_list.append(seg_len)
                    remaining -= seg_len
            
            seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32, device=self.device)
            seg_indptr = torch.zeros((num_segments + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
            
            if adapter_indices is None:
                weight_indices = torch.zeros((num_segments,), dtype=torch.int32, device=self.device)
            else:
                # Expand adapter indices for segments
                weight_indices = torch.repeat_interleave(
                    torch.tensor(adapter_indices, dtype=torch.int32, device=self.device),
                    num_segs_per_seq
                )
        
        if lora_ranks is None:
            lora_ranks_tensor = torch.full((num_adapters,), rank, dtype=torch.int64, device=self.device)
        else:
            lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int64, device=self.device)
        
        if scalings is None:
            scalings_tensor = torch.ones((num_adapters,), dtype=torch.float, device=self.device)
        else:
            scalings_tensor = torch.tensor(scalings, dtype=torch.float, device=self.device)
        
        permutation = torch.arange(total_tokens, dtype=torch.int32, device=self.device)
        
        batch_info = LoRABatchInfo(
            bs=bs,
            num_segments=num_segments,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            max_len=seq_len if segment_size is None else total_tokens,
            use_cuda_graph=False,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks_tensor,
            scalings=scalings_tensor,
            permutation=permutation,
        )
        
        # Get max rank for weight allocation
        max_rank = int(lora_ranks_tensor.max().item())
        
        # Create test inputs
        x_input = torch.randn((total_tokens, input_dim), device=self.device, dtype=self.dtype)
        
        # Create LoRA A weights (for shrink)
        lora_a_weights = torch.randn(
            (num_adapters, num_slices * max_rank, input_dim), 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Create LoRA B weights (for expand)
        lora_b_weights = torch.randn(
            (num_adapters, total_output_dim, max_rank),
            device=self.device,
            dtype=self.dtype
        )
        
        # Create slice offsets for expand operation
        if num_slices == 3:
            slice_offsets = torch.tensor(
                [0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
                dtype=torch.int32,
                device=self.device,
            )
            max_slice_size = max(q_dim, kv_dim)
        else:
            slice_offsets = torch.tensor(
                [0, q_dim] + [q_dim] * (num_slices - 1),
                dtype=torch.int32,
                device=self.device,
            )
            max_slice_size = q_dim
        
        # Test shrink operation
        shrink_output = chunked_sgmv_lora_shrink_forward(
            x=x_input,
            weights=lora_a_weights,
            batch_info=batch_info,
            num_slices=num_slices,
        )
        
        # Compute shrink reference using PyTorch
        expected_shrink = torch.zeros((total_tokens, num_slices * max_rank), device=self.device, dtype=self.dtype)
        
        if segment_size is None:
            # Simple case: one segment per sequence
            for seq_idx in range(bs):
                start_idx = seg_indptr[seq_idx].item()
                end_idx = seg_indptr[seq_idx + 1].item()
                adapter_idx = weight_indices[seq_idx].item()
                adapter_rank = lora_ranks_tensor[adapter_idx].item()
                
                if adapter_rank > 0:
                    expected_shrink[start_idx:end_idx, :num_slices * adapter_rank] = torch.mm(
                        x_input[start_idx:end_idx],
                        lora_a_weights[adapter_idx, :num_slices * adapter_rank].T
                    )
        else:
            # Chunked case: multiple segments per sequence
            for seg_idx in range(num_segments):
                start_idx = seg_indptr[seg_idx].item()
                end_idx = seg_indptr[seg_idx + 1].item()
                adapter_idx = weight_indices[seg_idx].item()
                adapter_rank = lora_ranks_tensor[adapter_idx].item()
                
                if adapter_rank > 0:
                    expected_shrink[start_idx:end_idx, :num_slices * adapter_rank] = torch.mm(
                        x_input[start_idx:end_idx],
                        lora_a_weights[adapter_idx, :num_slices * adapter_rank].T
                    )
        
        # Verify shrink operation
        self.assertEqual(shrink_output.shape, (total_tokens, num_slices * max_rank))
        torch.testing.assert_close(shrink_output, expected_shrink, atol=atol, rtol=rtol)
        
        # Test expand operation
        base_output = None
        if with_base_output:
            base_output = torch.randn((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
            original_base = base_output.clone()
        
        expand_output = chunked_sgmv_lora_expand_forward(
            x=shrink_output,
            lora_weight_b=lora_b_weights,
            batch_info=batch_info,
            slice_offsets=slice_offsets,
            max_slice_size=max_slice_size,
            base_output=base_output,
        )
        
        # Compute expand reference using PyTorch
        expected_expand = torch.zeros((total_tokens, total_output_dim), device=self.device, dtype=self.dtype)
        
        if segment_size is None:
            # Simple case: one segment per sequence
            for seq_idx in range(bs):
                start_idx = seg_indptr[seq_idx].item()
                end_idx = seg_indptr[seq_idx + 1].item()
                adapter_idx = weight_indices[seq_idx].item()
                adapter_rank = lora_ranks_tensor[adapter_idx].item()
                scaling = scalings_tensor[adapter_idx].item()
                
                if adapter_rank > 0:
                    if num_slices == 3:
                        # Q/K/V projection
                        expected_expand[start_idx:end_idx, :q_dim] = torch.mm(
                            shrink_output[start_idx:end_idx, :adapter_rank],
                            lora_b_weights[adapter_idx, :q_dim, :adapter_rank].T
                        ) * scaling
                        expected_expand[start_idx:end_idx, q_dim:q_dim + kv_dim] = torch.mm(
                            shrink_output[start_idx:end_idx, adapter_rank:2 * adapter_rank],
                            lora_b_weights[adapter_idx, q_dim:q_dim + kv_dim, :adapter_rank].T
                        ) * scaling
                        expected_expand[start_idx:end_idx, q_dim + kv_dim:] = torch.mm(
                            shrink_output[start_idx:end_idx, 2 * adapter_rank:3 * adapter_rank],
                            lora_b_weights[adapter_idx, q_dim + kv_dim:, :adapter_rank].T
                        ) * scaling
                    else:
                        # Other projections
                        for slice_idx in range(num_slices):
                            slice_start = slice_idx * adapter_rank
                            slice_end = (slice_idx + 1) * adapter_rank
                            expected_expand[start_idx:end_idx, :q_dim] += torch.mm(
                                shrink_output[start_idx:end_idx, slice_start:slice_end],
                                lora_b_weights[adapter_idx, :q_dim, :adapter_rank].T
                            ) * scaling
        else:
            # Chunked case: multiple segments per sequence
            for seg_idx in range(num_segments):
                start_idx = seg_indptr[seg_idx].item()
                end_idx = seg_indptr[seg_idx + 1].item()
                adapter_idx = weight_indices[seg_idx].item()
                adapter_rank = lora_ranks_tensor[adapter_idx].item()
                scaling = scalings_tensor[adapter_idx].item()
                
                if adapter_rank > 0:
                    if num_slices == 3:
                        # Q/K/V projection
                        expected_expand[start_idx:end_idx, :q_dim] = torch.mm(
                            shrink_output[start_idx:end_idx, :adapter_rank],
                            lora_b_weights[adapter_idx, :q_dim, :adapter_rank].T
                        ) * scaling
                        expected_expand[start_idx:end_idx, q_dim:q_dim + kv_dim] = torch.mm(
                            shrink_output[start_idx:end_idx, adapter_rank:2 * adapter_rank],
                            lora_b_weights[adapter_idx, q_dim:q_dim + kv_dim, :adapter_rank].T
                        ) * scaling
                        expected_expand[start_idx:end_idx, q_dim + kv_dim:] = torch.mm(
                            shrink_output[start_idx:end_idx, 2 * adapter_rank:3 * adapter_rank],
                            lora_b_weights[adapter_idx, q_dim + kv_dim:, :adapter_rank].T
                        ) * scaling
                    else:
                        # Other projections
                        for slice_idx in range(num_slices):
                            slice_start = slice_idx * adapter_rank
                            slice_end = (slice_idx + 1) * adapter_rank
                            expected_expand[start_idx:end_idx, :q_dim] += torch.mm(
                                shrink_output[start_idx:end_idx, slice_start:slice_end],
                                lora_b_weights[adapter_idx, :q_dim, :adapter_rank].T
                            ) * scaling
        
        if with_base_output:
            expected_expand += original_base
            self.assertTrue(torch.equal(expand_output, base_output))
        
        # Verify expand operation
        self.assertEqual(expand_output.shape, (total_tokens, total_output_dim))
        torch.testing.assert_close(expand_output, expected_expand, atol=atol, rtol=rtol)
        
        return shrink_output, expand_output
    
    # Test cases that invoke the helper with different configurations
    
    def test_basic_qkv_projection(self):
        """Test basic Q/K/V projection with default parameters."""
        self._test_chunked_sgmv_ops(
            bs=4,
            seq_len=32,
            input_dim=2048,
            rank=64,
            q_dim=2048,
            kv_dim=256,
            num_slices=3,
        )
    
    def test_small_batch_high_rank(self):
        """Test with small batch size and high rank."""
        self._test_chunked_sgmv_ops(
            bs=1,
            seq_len=64,
            input_dim=4096,
            rank=128,
            q_dim=4096,
            kv_dim=512,
            num_slices=3,
        )
    
    def test_large_batch_low_rank(self):
        """Test with large batch size and low rank."""
        self._test_chunked_sgmv_ops(
            bs=16,
            seq_len=16,
            input_dim=1024,
            rank=16,
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
        )
    
    def test_gate_up_projection(self):
        """Test gate/up projection (2 slices)."""
        self._test_chunked_sgmv_ops(
            bs=4,
            seq_len=32,
            input_dim=2048,
            rank=32,
            q_dim=4096,  # Used as output dim for gate/up
            kv_dim=0,    # Not used for gate/up
            num_slices=2,
        )
    
    def test_with_segmentation(self):
        """Test with sequence segmentation."""
        self._test_chunked_sgmv_ops(
            bs=2,
            seq_len=48,
            input_dim=2048,
            rank=64,
            q_dim=2048,
            kv_dim=256,
            num_slices=3,
            segment_size=16,
        )
    
    def test_multiple_adapters(self):
        """Test with multiple LoRA adapters."""
        self._test_chunked_sgmv_ops(
            bs=4,
            seq_len=16,
            input_dim=1024,
            rank=32,  # Default, will be overridden by lora_ranks
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
            num_adapters=3,
            adapter_indices=[0, 1, 2, 1],  # Different adapters for different sequences
            lora_ranks=[32, 64, 16],       # Different ranks for each adapter
            scalings=[1.0, 0.5, 2.0],       # Different scalings
        )
    
    def test_with_base_output(self):
        """Test with base output accumulation."""
        self._test_chunked_sgmv_ops(
            bs=2,
            seq_len=16,
            input_dim=1024,
            rank=32,
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
            with_base_output=True,
        )
    
    def test_zero_rank_adapter(self):
        """Test with zero-rank adapter (no-op case)."""
        self._test_chunked_sgmv_ops(
            bs=2,
            seq_len=16,
            input_dim=1024,
            rank=0,
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
        )
    
    def test_mixed_zero_nonzero_ranks(self):
        """Test with mix of zero and non-zero rank adapters."""
        self._test_chunked_sgmv_ops(
            bs=4,
            seq_len=16,
            input_dim=1024,
            rank=32,
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
            num_adapters=2,
            adapter_indices=[0, 1, 0, 1],
            lora_ranks=[0, 32],  # First adapter has zero rank
            scalings=[1.0, 0.5],
        )
    
    def test_uneven_qkv_dimensions(self):
        """Test with very uneven Q/K/V dimensions."""
        self._test_chunked_sgmv_ops(
            bs=2,
            seq_len=32,
            input_dim=2048,
            rank=48,
            q_dim=8192,  # Very large Q
            kv_dim=64,   # Very small K/V
            num_slices=3,
        )
    
    def test_single_token_sequence(self):
        """Test with single token sequences."""
        self._test_chunked_sgmv_ops(
            bs=8,
            seq_len=1,
            input_dim=1024,
            rank=32,
            q_dim=1024,
            kv_dim=128,
            num_slices=3,
        )
    
    def test_large_sequence_length(self):
        """Test with large sequence length."""
        self._test_chunked_sgmv_ops(
            bs=1,
            seq_len=512,
            input_dim=2048,
            rank=64,
            q_dim=2048,
            kv_dim=256,
            num_slices=3,
            segment_size=64,
            atol=1e-1,  # Relax tolerance for larger sequences
            rtol=1e-1,
        )
    
    def test_power_of_two_dimensions(self):
        """Test with all power-of-two dimensions."""
        self._test_chunked_sgmv_ops(
            bs=8,
            seq_len=64,
            input_dim=2048,
            rank=64,
            q_dim=4096,
            kv_dim=512,
            num_slices=3,
        )
    
    def test_non_power_of_two_dimensions(self):
        """Test with non-power-of-two dimensions."""
        self._test_chunked_sgmv_ops(
            bs=3,
            seq_len=23,
            input_dim=1536,
            rank=37,
            q_dim=3072,
            kv_dim=384,
            num_slices=3,
        )


if __name__ == "__main__":
    unittest.main()