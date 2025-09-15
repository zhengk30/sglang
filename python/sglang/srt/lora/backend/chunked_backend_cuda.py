from typing import Optional

import torch

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Try to import CUDA ops, fall back to Triton if not available
try:
    from sglang.srt.lora.cuda_ops import (
        chunked_sgmv_lora_expand_forward,
        chunked_sgmv_lora_shrink_forward,
    )
    USE_CUDA = True
except ImportError:
    from sglang.srt.lora.triton_ops import (
        chunked_sgmv_lora_expand_forward,
        chunked_sgmv_lora_shrink_forward,
    )
    USE_CUDA = False


class ChunkedSgmvLoRABackendCUDA(ChunkedSgmvLoRABackend):
    """
    Chunked LoRA backend using CUDA kernels for segmented matrix-vector multiplication.

    This backend provides a CUDA implementation of the SGMV algorithm, offering
    potentially better performance than the Triton version for certain workloads.
    Falls back to Triton if CUDA kernels are not available.
    """

    name = "csgmv_cuda"

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        super().__init__(max_loras_per_batch, device)
        self.segment_size = 16 
        self.use_cuda = USE_CUDA
        if self.use_cuda:
            print("Using CUDA kernels for chunked SGMV LoRA backend")
        else:
            print("CUDA kernels not available, falling back to Triton for chunked SGMV LoRA backend")

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return chunked_sgmv_lora_shrink_forward(
            x,
            weights,
            self.batch_info,
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        # For simple lora B, we use slice offsets [0, output_dim]
        output_dim = weights.shape[-2]
        max_slice_size = output_dim
        return chunked_sgmv_lora_expand_forward(
            x=x,
            lora_weight_b=weights,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_slice_size,
            base_output=base_output,
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x,
            qkv_lora_a,
            self.batch_info,
            num_slices=3,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            lora_weight_b=qkv_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_qkv_out_dim,
            base_output=base_output,
        )
        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        assert isinstance(gate_up_lora_b, torch.Tensor)
        output_dim = gate_up_lora_b.shape[-2] // 2

        # lora_a_output: (s, 2 * r)
        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x,
            gate_up_lora_a,
            self.batch_info,
            num_slices=2,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            lora_weight_b=gate_up_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=output_dim,
            base_output=base_output,
        )
        return lora_output

