import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the exact pattern from Mahmoud8 graphs with slice to 128 and expand to (1, 128).
    """
    tmp_2 = in_1[slice(None, None, None), slice(None, 128, None)]
    tmp_3 = tmp_2.expand(1, 128)
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1):
    """Extract arguments needed for replacement."""
    return (in_0, in_1)


@triton.jit
def copy_slice_kernel(
    in_ptr,
    out_ptr,
    slice_size,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple kernel to copy a slice efficiently.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < slice_size
    
    # Load and store with coalescing
    data = tl.load(in_ptr + offsets * stride_in, mask=mask, other=0)
    tl.store(out_ptr + offsets * stride_out, data, mask=mask)


@torch.fx.wrap
def optimized_slice_expand_unsqueeze(in_0, in_1):
    """
    Optimized implementation using Triton kernel for slice operation.
    The unsqueeze is kept as-is since it's just a view.
    """
    slice_size = 128
    
    # Optimize the slice operation using Triton
    # For int64 tensors, we'll use a simple copy kernel
    tmp_3 = torch.empty((1, slice_size), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(slice_size, BLOCK_SIZE),)
    
    copy_slice_kernel[grid](
        in_1,
        tmp_3,
        slice_size,
        in_1.stride(1),
        tmp_3.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Unsqueeze operation - this is just a view, keep it efficient
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    
    return (tmp_3, tmp_4)


def replacement_func():
    """Return the replacement function."""
    return optimized_slice_expand_unsqueeze