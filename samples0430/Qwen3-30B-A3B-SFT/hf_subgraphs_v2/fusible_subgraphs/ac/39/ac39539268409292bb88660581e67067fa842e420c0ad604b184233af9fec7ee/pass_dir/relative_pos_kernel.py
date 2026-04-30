import torch
import triton
import triton.language as tl


@triton.jit
def relative_pos_bias_kernel(
    out_ptr,
    grid_size,
    offset_val,
    multiplier_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Directly computes the position bias table.
    For flat index `idx` representing position `row, col`:
        row = idx // (grid_size + 1)
        col = idx %  (grid_size + 1)
        out = (row - offset) * multiplier + (col - offset)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert flat index to 2D coordinates in the (grid+1) x (grid+1) table
    row = offsets // (grid_size + 1)
    col = offsets % (grid_size + 1)

    # Compute position bias index
    out = (row - offset_val) * multiplier_val + (col - offset_val)

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def get_relative_pos_table_flat(grid_size: int, offset_val: int, multiplier_val: int) -> torch.Tensor:
    """
    Returns the flattened relative position bias table as a 1-D int64 tensor.
    The table is of shape (grid_size+1)^2.
    """
    n_elements = (grid_size + 1) * (grid_size + 1)
    out = torch.empty(n_elements, dtype=torch.int64)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    relative_pos_bias_kernel[grid](
        out,
        grid_size,
        offset_val,
        multiplier_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out