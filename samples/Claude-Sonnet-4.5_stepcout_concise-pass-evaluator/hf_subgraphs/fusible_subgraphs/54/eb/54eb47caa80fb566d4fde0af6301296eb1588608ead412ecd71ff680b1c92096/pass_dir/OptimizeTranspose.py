import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match: simple transpose operation
    """
    tmp_5 = in_0.transpose(0, 1)
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


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
    Optimized transpose kernel for swapping dimensions 0 and 1
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


@torch.fx.wrap
def optimized_transpose(in_0):
    """
    Wrapper function for optimized transpose
    """
    dim0, dim1, dim2 = in_0.shape
    out_shape = (dim1, dim0, dim2)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
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
    
    return out


def replacement_func():
    return optimized_transpose