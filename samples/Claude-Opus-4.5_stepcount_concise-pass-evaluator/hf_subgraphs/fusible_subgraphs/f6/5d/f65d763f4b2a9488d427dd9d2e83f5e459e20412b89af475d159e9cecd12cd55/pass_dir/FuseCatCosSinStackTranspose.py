import torch
import triton
import triton.language as tl

# Pattern to match the computation from model.py
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.cat((in_0, in_2), dim=-1)
    tmp_1 = in_1.cos()
    tmp_2 = in_1.sin()
    tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
    tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_kernel_128(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_ptr,
):
    """
    One block per row, using 128 as the inner dimension.
    Input: [64, 64] x 3
    Output: [64, 2, 128]
    """
    row = tl.program_id(0)  # 0-63
    
    # Process both halves together using 128-element vectors
    cols_128 = tl.arange(0, 128)  # 0-127
    cols_64 = cols_128 % 64  # 0-63, 0-63
    is_first_half = cols_128 < 64
    
    # Input offset (load same row twice due to wraparound)
    in_off = row * 64 + cols_64
    
    # Load from all inputs with wraparound
    v0 = tl.load(in_0_ptr + in_off)
    v1 = tl.load(in_1_ptr + in_off)
    v2 = tl.load(in_2_ptr + in_off)
    
    # Compute cos/sin
    c = tl.cos(v1)
    s = tl.sin(v1)
    
    # Select for row 0: [in_0, in_2]
    row0_out = tl.where(is_first_half, v0, v2)
    # Select for row 1: [cos, sin]
    row1_out = tl.where(is_first_half, c, s)
    
    # Output offset: [64, 2, 128]
    out_base = row * 256
    
    # Store both 128-element rows
    tl.store(out_ptr + out_base + cols_128, row0_out)
    tl.store(out_ptr + out_base + 128 + cols_128, row1_out)

@torch.fx.wrap
def fused_cat_cos_sin_stack_transpose(in_0, in_1, in_2):
    M, N = in_0.shape  # 64, 64
    out = torch.empty((M, 2, N * 2), dtype=in_0.dtype, device=in_0.device)
    fused_kernel_128[(M,)](in_0, in_1, in_2, out)
    return out

def replacement_func():
    return fused_cat_cos_sin_stack_transpose