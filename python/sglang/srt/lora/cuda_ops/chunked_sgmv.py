import torch
from typing import Optional

try:
    import sgl_kernel
    # Check if the ops are available
    CUDA_OPS_AVAILABLE = hasattr(torch.ops.sgl_kernel, 'chunked_sgmv_lora_shrink')
except ImportError:
    CUDA_OPS_AVAILABLE = False

from sglang.srt.lora.utils import LoRABatchInfo


def chunked_sgmv_lora_shrink_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int = 1,
) -> torch.Tensor:
    """
    CUDA implementation of chunked SGMV LoRA shrink operation.

    This function provides a drop-in replacement for the Triton version,
    maintaining the same interface and behavior.

    Args:
        x: (s, input_dim) - Input activations
        weights: (num_lora, num_slices * r, input_dim) - LoRA A weights
        batch_info: Batch information containing segment info and permutations
        num_slices: Number of slices (3 for QKV, 2 for gate_up, 1 for others)

    Returns:
        output: (s, num_slices * r) - Intermediate activations
    """
    if not CUDA_OPS_AVAILABLE:
        raise ImportError("CUDA ops not available. Please rebuild sgl-kernel.")

    assert x.is_contiguous(), "x must be contiguous"
    assert weights.is_contiguous(), "weights must be contiguous"
    assert len(x.shape) == 2, "x must be 2D"
    assert len(weights.shape) == 3, "weights must be 3D"

    # Call the CUDA kernel
    output = torch.ops.sgl_kernel.chunked_sgmv_lora_shrink(
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.num_segments,
        num_slices,
    )

    return output


def chunked_sgmv_lora_expand_forward(
    x: torch.Tensor,
    lora_weight_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CUDA implementation of chunked SGMV LoRA expand operation.

    This function provides a drop-in replacement for the Triton version,
    maintaining the same interface and behavior.

    Args:
        x: (s, slice_num * r) - Intermediate activations from LoRA A
        lora_weight_b: (num_lora, output_dim, r) - LoRA B weights
        batch_info: Batch information containing segment info and permutations
        slice_offsets: Boundaries for different slices in the output dimension
        max_slice_size: Maximum size of any slice
        base_output: Optional base output to accumulate onto

    Returns:
        output: (s, output_dim) - Final output with LoRA applied
    """
    if not CUDA_OPS_AVAILABLE:
        raise ImportError("CUDA ops not available. Please rebuild sgl-kernel.")

    assert x.is_contiguous(), "x must be contiguous"
    assert lora_weight_b.is_contiguous(), "lora_weight_b must be contiguous"

    # Get dimensions
    s = x.shape[0]
    output_dim = lora_weight_b.shape[1]

    # Initialize output if not provided
    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    # Call the CUDA kernel
    output = torch.ops.sgl_kernel.chunked_sgmv_lora_expand(
        x,
        lora_weight_b,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        slice_offsets,
        batch_info.num_segments,
        max_slice_size,
        output if base_output is not None else None,
    )

    return output