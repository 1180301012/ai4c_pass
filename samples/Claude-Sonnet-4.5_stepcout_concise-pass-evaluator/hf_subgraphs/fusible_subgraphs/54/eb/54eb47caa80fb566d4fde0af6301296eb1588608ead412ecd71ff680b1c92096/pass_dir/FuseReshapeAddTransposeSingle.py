import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the complete pattern including both uses of reshape
    """
    tmp_0 = in_1.reshape(1, 64, -1)
    tmp_1 = in_0 + tmp_0
    tmp_2 = in_0 + tmp_0  # Both uses must be in pattern
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)
    tmp_5 = in_0.transpose(0, 1)
    return tmp_4, tmp_3, tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_reshape_add_transpose_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    dim0,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Reads in_1 with reshape indexing [64, 2, 128] -> [1, 64, 256]
    2. Adds in_0 + reshaped_in_1 with broadcasting
    3. Writes transposed result (swap dim0 and dim1)
    """
    # Calculate which element this thread handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Total elements in output
    n_elements = dim0 * dim1 * dim2
    mask = offsets < n_elements
    
    # Compute original indices (before transpose) for the addition result
    # Output indices are for transposed shape: [dim1, dim0, dim2]
    idx_2 = offsets % dim2
    idx_1 = (offsets // dim2) % dim0
    idx_0 = offsets // (dim2 * dim0)
    
    # Original indices before transpose: swap back
    orig_idx_0 = idx_1
    orig_idx_1 = idx_0
    orig_idx_2 = idx_2
    
    # Linear index in original layout
    orig_offset = orig_idx_0 * (dim1 * dim2) + orig_idx_1 * dim2 + orig_idx_2
    
    # Load in_0
    in_0_val = tl.load(in_0_ptr + orig_offset, mask=mask, other=0.0)
    
    # Compute in_1 index with reshape [64, 2, 128] -> [1, 64, 256]
    # orig_idx_0 corresponds to dimension that becomes 1 (broadcasts)
    # orig_idx_1 corresponds to dimension 64
    # orig_idx_2 corresponds to dimension 256 = 2 * 128
    
    # Map back to original in_1 shape [64, 2, 128]
    in1_idx_0 = orig_idx_1  # maps to 64
    in1_idx_1 = orig_idx_2 // 128  # maps to 2
    in1_idx_2 = orig_idx_2 % 128   # maps to 128
    
    in1_offset = in1_idx_0 * (2 * 128) + in1_idx_1 * 128 + in1_idx_2
    in_1_val = tl.load(in_1_ptr + in1_offset, mask=mask, other=0.0)
    
    # Compute addition
    add_result = in_0_val + in_1_val
    
    # Store transposed addition result
    tl.store(out_ptr + offsets, add_result, mask=mask)


@torch.fx.wrap
def fused_reshape_add_transpose(in_0, in_1):
    """
    Wrapper function that launches the fused kernel
    Returns tmp_4, tmp_3, tmp_5
    """
    # Get dimensions
    dim0, dim1, dim2 = in_0.shape
    
    # Output shapes after transpose(0, 1)
    out_shape = (dim1, dim0, dim2)
    
    # Create output tensors - need separate objects for tmp_3 and tmp_4
    out_4 = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    out_3 = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    out_5 = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel for tmp_4
    n_elements = dim0 * dim1 * dim2
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_reshape_add_transpose_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out_4,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch kernel for tmp_3 (same computation as tmp_4)
    fused_reshape_add_transpose_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out_3,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch kernel for tmp_5 (transpose of in_0)
    transpose_kernel[grid](
        in_ptr=in_0,
        out_ptr=out_5,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return tmp_4, tmp_3, tmp_5 as separate objects
    return out_4, out_3, out_5


@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    dim0,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose kernel for swapping dimensions 0 and 1
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    n_elements = dim0 * dim1 * dim2
    mask = offsets < n_elements
    
    # Output indices are for transposed shape: [dim1, dim0, dim2]
    idx_2 = offsets % dim2
    idx_1 = (offsets // dim2) % dim0
    idx_0 = offsets // (dim2 * dim0)
    
    # Original indices before transpose: swap back
    orig_idx_0 = idx_1
    orig_idx_1 = idx_0
    orig_idx_2 = idx_2
    
    # Linear index in original layout
    orig_offset = orig_idx_0 * (dim1 * dim2) + orig_idx_1 * dim2 + orig_idx_2
    
    # Load and store
    val = tl.load(in_ptr + orig_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def replacement_func():
    return fused_reshape_add_transpose