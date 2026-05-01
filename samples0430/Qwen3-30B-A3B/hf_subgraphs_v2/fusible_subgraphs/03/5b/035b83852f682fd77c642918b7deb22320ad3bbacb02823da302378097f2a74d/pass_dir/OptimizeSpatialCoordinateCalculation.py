import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern for coordinate calculation: arange(14), view, subtraction, repeats, etc.
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_19 = tmp_9.unsqueeze(0)
    return tmp_15, tmp_17, tmp_19

def replacement_args(in_0, in_1, in_2):
    # Return inputs needed for replacement (the coordinate calculation doesn't depend on inputs)
    return (in_0, in_1, in_2)

@triton.jit
def compute_coordinates_kernel(
    x_out_ptr,
    y_out_ptr,
    dist2_out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    
    x_vals = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    y_vals = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    dist2_vals = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            row_idx = row_offsets[i]
            col_idx = col_offsets[j]
            
            # Convert 1D index to 2D grid (14x14)
            i1 = row_idx // 14
            i2 = row_idx % 14
            j1 = col_idx // 14
            j2 = col_idx % 14
            
            dx = i1 - j1
            dy = i2 - j2
            
            x_vals[i, j] = dx
            y_vals[i, j] = dy
            dist2_vals[i, j] = dx*dx + dy*dy
    
    tl.store(x_out_ptr + row_start * n_cols + col_start, x_vals, mask=row_mask[:, None] & col_mask[None, :])
    tl.store(y_out_ptr + row_start * n_cols + col_start, y_vals, mask=row_mask[:, None] & col_mask[None, :])
    tl.store(dist2_out_ptr + row_start * n_cols + col_start, dist2_vals, mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def compute_coordinates(in_0, in_1, in_2):
    n_rows = 196
    n_cols = 196
    BLOCK_SIZE = 32
    grid = (n_rows // BLOCK_SIZE, n_cols // BLOCK_SIZE)
    
    x_out = torch.empty((1, n_rows, n_cols), dtype=in_2.dtype, device=in_2.device)
    y_out = torch.empty((1, n_rows, n_cols), dtype=in_2.dtype, device=in_2.device)
    dist2_out = torch.empty((1, n_rows, n_cols), dtype=in_2.dtype, device=in_2.device)
    
    compute_coordinates_kernel[grid](
        x_out, y_out, dist2_out,
        n_rows, n_cols,
        BLOCK_SIZE
    )
    
    # Stack channels into (1, 196, 196, 3) - order: [channel0, channel1, channel2]
    # Original: channel0 = x, channel1 = y, channel2 = dist2
    out = torch.empty((1, 196, 196, 3), dtype=in_2.dtype, device=in_2.device)
    out[..., 0] = x_out
    out[..., 1] = y_out
    out[..., 2] = dist2_out
    return out

def replacement_func():
    return compute_coordinates