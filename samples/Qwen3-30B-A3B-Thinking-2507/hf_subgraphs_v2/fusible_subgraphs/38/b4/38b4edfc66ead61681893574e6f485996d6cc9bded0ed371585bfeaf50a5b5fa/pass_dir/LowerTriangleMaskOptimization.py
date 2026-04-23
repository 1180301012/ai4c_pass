import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(n, device):
    tmp_1 = torch.full((n, n), -3.4028234663852886e+38, device=device)
    tmp_2 = torch.arange(n, device=device)
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(n, 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    return tmp_6

# Argument extraction function
def replacement_args(n, device):
    return (n, device)

# Triton kernel
def lower_triangle_mask_kernel(out_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)
    offsets_x = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_y = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_x = offsets_x < n
    mask_y = offsets_y < n
    mask = mask_x[:, None] & mask_y[None, :]
    
    row = offsets_x[:, None]
    col = offsets_y[None, :]
    
    is_upper = row < col
    val = tl.where(is_upper, 0.0, float('-inf'))
    
    tl.store(out_ptr + row * n + col, val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def lower_triangle_mask_kernel_wrapper(n, device):
    out = torch.empty((n, n), device=device, dtype=torch.float32)
    grid_x = (n + 31) // 32
    grid_y = (n + 31) // 32
    lower_triangle_mask_kernel[(grid_x, grid_y)](out, n, BLOCK_SIZE=32)
    return out

# Replacement function
def replacement_func():
    return lower_triangle_mask_kernel_wrapper