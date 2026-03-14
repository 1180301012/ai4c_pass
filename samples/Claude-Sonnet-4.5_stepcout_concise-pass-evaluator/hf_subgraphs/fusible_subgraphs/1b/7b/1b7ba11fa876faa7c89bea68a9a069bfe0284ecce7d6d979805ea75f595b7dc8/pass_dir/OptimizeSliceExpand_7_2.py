import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the exact pattern from BAAI_AltCLIP graph with slice to 7 and expand to (2, 7).
    """
    tmp_2 = in_1[slice(None, None, None), slice(None, 7, None)]
    tmp_3 = tmp_2.expand(2, 7)
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1):
    """Extract arguments needed for replacement."""
    return (in_0, in_1)


@triton.jit
def slice_expand_broadcast_kernel(
    in_ptr,
    out_ptr,
    slice_size,
    expand_rows,
    stride_in,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to slice and expand (broadcast) data efficiently.
    """
    pid = tl.program_id(0)
    row = pid // ((slice_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    col_block = pid % ((slice_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    block_start = col_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (row < expand_rows) & (offsets < slice_size)
    
    # Load from input (always from row 0)
    data = tl.load(in_ptr + offsets * stride_in, mask=(offsets < slice_size), other=0)
    
    # Store to output at the appropriate row
    out_idx = row * stride_out_row + offsets * stride_out_col
    tl.store(out_ptr + out_idx, data, mask=mask)


@torch.fx.wrap
def optimized_slice_expand_7_2(in_0, in_1):
    """
    Optimized implementation for slice to 7 and expand to (2, 7).
    """
    slice_size = 7
    expand_rows = 2
    
    # Create output tensor
    tmp_3 = torch.empty((expand_rows, slice_size), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 64  # Small block size for small data
    grid_cols = triton.cdiv(slice_size, BLOCK_SIZE)
    grid = (expand_rows * grid_cols,)
    
    slice_expand_broadcast_kernel[grid](
        in_1,
        tmp_3,
        slice_size,
        expand_rows,
        in_1.stride(1),
        tmp_3.stride(0),
        tmp_3.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Unsqueeze operation - view only
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    
    return (tmp_3, tmp_4)


def replacement_func():
    """Return the replacement function."""
    return optimized_slice_expand_7_2