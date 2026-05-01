import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def sum_reduce_kernel(
    X_ptr,
    Sum_ptr,
    B,
    H,
    S,
    BLOCK_SIZE: tl.constexpr
):
    # Each block computes one row sum (across last dimension)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    sum_val = tl.zeros((1,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        pos = s * BLOCK_SIZE + i
        if pos < S:
            x = tl.load(X_ptr + b * H * S * S + h * S * S + s * S + pos)
            sum_val += x
    tl.store(Sum_ptr + b * H * S + h * S + s, sum_val)

@triton.jit
def div_kernel(
    X_ptr,
    Sum_ptr,
    Y_ptr,
    B,
    H,
    S,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (B, H, S) blocks
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    l = tl.thread_id(0)
    if l < S:
        x = tl.load(X_ptr + b * H * S * S + h * S * S + s * S + l)
        sum_val = tl.load(Sum_ptr + b * H * S + h * S + s)
        y = x / sum_val
        tl.store(Y_ptr + b * H * S * S + h * S * S + s * S + l, y)

@torch.fx.wrap
def normalize_sum_div(in_0):
    B, H, S, _ = in_0.shape
    sum_buffer = torch.empty((B, H, S), device=in_0.device, dtype=in_0.dtype)
    
    # BLOCK_SIZE tuned for input shape
    BLOCK_SIZE = 128
    
    # Launch sum kernel (3D grid: B x H x S)
    sum_reduce_kernel[(B, H, S)](
        in_0, sum_buffer, B, H, S, BLOCK_SIZE
    )
    
    out = torch.empty_like(in_0)
    
    # Launch div kernel (3D grid: B x H x S)
    div_kernel[(B, H, S)](
        in_0, sum_buffer, out, B, H, S, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return normalize_sum_div