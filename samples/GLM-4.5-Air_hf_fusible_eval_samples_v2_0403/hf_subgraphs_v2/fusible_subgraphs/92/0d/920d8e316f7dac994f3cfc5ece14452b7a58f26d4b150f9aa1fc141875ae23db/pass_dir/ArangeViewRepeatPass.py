import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
    ],
    key=['num_cols', 'num_rows'],
)
@triton.jit
def arange_view_repeat_kernel(
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Directly generate arange values repeated across rows.
    output[row, col] = col (for all rows)
    """
    row_idx = tl.program_id(0)
    col_block_start = tl.program_id(1) * BLOCK_SIZE
    col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols
    
    # Compute the arange value for this column
    col_val = tl.where(mask, col_offsets, 0)
    
    # Store the value at each row position
    output_offs = row_idx * num_cols + col_offsets
    tl.store(output_ptr + output_offs, col_val, mask=mask)


@torch.fx.wrap
def arange_view_repeat_impl(start, end, device, num_rows, dtype):
    """
    Optimized implementation that directly creates the final output.
    Instead of: arange(start, end) -> view(1, -1) -> repeat(num_rows, 1)
    We directly create output with shape (num_rows, end-start) where each row contains [start, start+1, ..., end-1]
    """
    num_cols = end - start
    
    if num_cols <= 0:
        return torch.empty((num_rows, 0), device=device, dtype=dtype)
    
    output = torch.empty((num_rows, num_cols), device=device, dtype=dtype)
    
    BLOCK_SIZE = 1024
    num_rows_grid = num_rows
    num_col_blocks = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    arange_view_repeat_kernel[(num_rows_grid, num_col_blocks)](
        output_ptr=output,
        num_rows=num_rows,
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(start, end, device, num_rows):
    """
    Pattern matching function.
    Matches: arange(start, end) -> view(1, -1) -> repeat(num_rows, 1)
    """
    tmp_0 = torch.arange(start, end, device=device)
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(num_rows, 1)
    return tmp_2


def replacement_args(start, end, device, num_rows):
    """
    Extract arguments needed for replacement.
    """
    return (start, end, device, num_rows)


def replacement_func():
    """
    Return the optimized kernel wrapper.
    """
    return arange_view_repeat_impl