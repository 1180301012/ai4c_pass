import torch
import triton
import triton.language as tl


@triton.jit
def scale_transpose_kernel(
    x_ptr,
    output_ptr,
    scale,
    n_batch,
    n_heads,
    n_rows,
    n_cols,
    stride_x_b,
    stride_x_h,
    stride_x_row,
    stride_x_col,
    stride_out_b,
    stride_out_h,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for scale + transpose.
    Each program handles BLOCK_SIZE elements in a row.
    """
    # Get indices
    matrix_row_idx = tl.program_id(0)
    col_tile_idx = tl.program_id(1)
    
    # Decode batch, head, row
    matrices_per_batch = n_heads * n_rows
    batch_idx = matrix_row_idx // matrices_per_batch
    remainder = matrix_row_idx % matrices_per_batch
    head_idx = remainder // n_rows
    row_idx = remainder % n_rows
    
    # Bounds check
    if batch_idx >= n_batch or head_idx >= n_heads or row_idx >= n_rows:
        return
    
    # Compute base pointers
    x_base = batch_idx * stride_x_b + head_idx * stride_x_h + row_idx * stride_x_row
    out_base = batch_idx * stride_out_b + head_idx * stride_out_h
    
    # Compute column offset for this tile
    col_offset = col_tile_idx * BLOCK_SIZE
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    # Load and scale
    x_offsets = x_base + col_offsets * stride_x_col
    tile = tl.load(x_ptr + x_offsets, mask=col_mask, other=0.0) * scale
    
    # Store transposed: output[col, row] = scaled_input[row, col]
    out_offsets = out_base + col_offsets * stride_out_row + row_idx * stride_out_col
    tl.store(output_ptr + out_offsets, tile, mask=col_mask)


@torch.fx.wrap
def scale_transpose(x, scale):
    """
    Fused kernel wrapper: scale + transpose.
    """
    batch, heads, n_rows, n_cols = x.shape
    
    # Output: [batch, heads, n_cols, n_rows]
    output = torch.empty(batch, heads, n_cols, n_rows, dtype=x.dtype, device=x.device)
    
    # Grid: (batch * heads * n_rows, n_col_tiles)
    BLOCK_SIZE = 128
    num_rows = batch * heads * n_rows
    num_col_tiles = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_rows, num_col_tiles)
    
    scale_transpose_kernel[grid](
        x,
        output,
        scale,
        batch,
        heads,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_SIZE,
    )
    
    return output


@triton.jit
def scale_softmax_transpose_kernel(
    x_ptr,
    output_ptr,
    scale,
    n_batch,
    n_heads,
    n_rows,
    n_cols,
    stride_x_b,
    stride_x_h,
    stride_x_row,
    stride_x_col,
    stride_out_b,
    stride_out_h,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for scale + softmax + transpose.
    Each program handles one complete row and computes softmax on it.
    BLOCK_SIZE must be >= n_cols.
    """
    # Get indices
    matrix_row_idx = tl.program_id(0)
    
    # Decode batch, head, row
    matrices_per_batch = n_heads * n_rows
    batch_idx = matrix_row_idx // matrices_per_batch
    remainder = matrix_row_idx % matrices_per_batch
    head_idx = remainder // n_rows
    row_idx = remainder % n_rows
    
    # Bounds check
    if batch_idx >= n_batch or head_idx >= n_heads or row_idx >= n_rows:
        return
    
    # Compute base pointers
    x_base = batch_idx * stride_x_b + head_idx * stride_x_h + row_idx * stride_x_row
    out_base = batch_idx * stride_out_b + head_idx * stride_out_h
    
    # Load the complete row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    x_offsets = x_base + col_offsets * stride_x_col
    row_data = tl.load(x_ptr + x_offsets, mask=col_mask, other=0.0) * scale
    
    # Apply softmax using Triton's built-in softmax (dim=0 means softmax over the row)
    # But we need softmax over cols, which is dim=-1. In this 1D array, dim=0.
    softmax_row = tl.softmax(row_data, dim=0)
    
    # Store transposed
    out_offsets = out_base + col_offsets * stride_out_row + row_idx * stride_out_col
    tl.store(output_ptr + out_offsets, softmax_row, mask=col_mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(x, scale):
    """
    Complete fused kernel: scale + softmax + transpose.
    Uses a single Triton kernel that processes complete rows.
    """
    batch, heads, n_rows, n_cols = x.shape
    
    # Output: [batch, heads, n_cols, n_rows]
    output = torch.empty(batch, heads, n_cols, n_rows, dtype=x.dtype, device=x.device)
    
    # Grid: one program per row
    BLOCK_SIZE = 512  # Large enough for n_cols=400
    num_rows = batch * heads * n_rows
    grid = (num_rows,)
    
    scale_softmax_transpose_kernel[grid](
        x,
        output,
        scale,
        batch,
        heads,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_0):
    """
    Match the pattern: scale * 0.1767 -> softmax -> transpose
    """
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0, 0.1767766952966369)


def replacement_func():
    return fused_scale_softmax_transpose