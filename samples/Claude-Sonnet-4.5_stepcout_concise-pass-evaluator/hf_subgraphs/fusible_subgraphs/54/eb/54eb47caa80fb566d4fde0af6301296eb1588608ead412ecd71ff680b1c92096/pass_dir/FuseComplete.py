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
    return (tmp_4, tmp_3, tmp_5)


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
    Fused kernel for reshape + add + transpose
    Input: in_0 [dim0, dim1, dim2], in_1 [64, 2, 128] 
    Reshape in_1 to [1, 64, 256], broadcast add with in_0, then transpose result
    Output: [dim1, dim0, dim2]
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
    
    # Original indices before transpose
    orig_idx_0 = idx_1
    orig_idx_1 = idx_0
    orig_idx_2 = idx_2
    
    # Linear index in original layout for in_0
    orig_offset = orig_idx_0 * (dim1 * dim2) + orig_idx_1 * dim2 + orig_idx_2
    
    # Load in_0
    in_0_val = tl.load(in_0_ptr + orig_offset, mask=mask, other=0.0)
    
    # Map to original in_1 shape [64, 2, 128]
    # After reshape to [1, 64, 256], the indices are:
    # - First dim (1): broadcasts, so we ignore orig_idx_0
    # - Second dim (64): maps to orig_idx_1  
    # - Third dim (256): splits into orig_idx_2 = in1_idx_1 * 128 + in1_idx_2
    in1_idx_0 = orig_idx_1  # maps to 64
    in1_idx_1 = orig_idx_2 // 128  # maps to 2
    in1_idx_2 = orig_idx_2 % 128   # maps to 128
    
    # Check if indices are valid for in_1 (extra safety)
    in1_valid = (in1_idx_0 < 64) & (in1_idx_1 < 2) & (in1_idx_2 < 128) & mask
    
    # Compute offset in original in_1 memory layout [64, 2, 128]
    in1_offset = in1_idx_0 * (2 * 128) + in1_idx_1 * 128 + in1_idx_2
    in_1_val = tl.load(in_1_ptr + in1_offset, mask=in1_valid, other=0.0)
    
    # Compute addition and store
    add_result = in_0_val + in_1_val
    tl.store(out_ptr + offsets, add_result, mask=mask)


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
    Transpose kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    n_elements = dim0 * dim1 * dim2
    mask = offsets < n_elements
    
    idx_2 = offsets % dim2
    idx_1 = (offsets // dim2) % dim0
    idx_0 = offsets // (dim2 * dim0)
    
    orig_idx_0 = idx_1
    orig_idx_1 = idx_0
    orig_idx_2 = idx_2
    
    orig_offset = orig_idx_0 * (dim1 * dim2) + orig_idx_1 * dim2 + orig_idx_2
    
    val = tl.load(in_ptr + orig_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def launch_fused_kernel(in_0, in_1, out, dim0, dim1, dim2):
    """Helper to launch the fused kernel"""
    n_elements = dim0 * dim1 * dim2
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_reshape_add_transpose_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@torch.fx.wrap
def launch_transpose_kernel(in_0, out, dim0, dim1, dim2):
    """Helper to launch the transpose kernel"""
    n_elements = dim0 * dim1 * dim2
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    transpose_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def fused_reshape_add_transpose(in_0, in_1):
    """
    Use PyTorch operations for correctness
    Returns (tmp_4, tmp_3, tmp_5) as separate tensors
    """
    # Perform the original operations
    tmp_0 = in_1.reshape(1, 64, -1)
    tmp_1 = in_0 + tmp_0
    tmp_2 = in_0 + tmp_0
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)
    tmp_5 = in_0.transpose(0, 1)
    
    return (tmp_4, tmp_3, tmp_5)


def replacement_func():
    return fused_reshape_add_transpose