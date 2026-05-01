import torch
import triton
import triton.language as tl

def pattern(x):
    t9 = x
    t10 = t9.unsqueeze(2)
    t11 = t9.unsqueeze(3)
    t12 = t10 - t11
    t13 = t12 != 0
    t14 = t12.masked_fill(t13, -1000.0)
    t15 = t12 == 0
    t16 = t14.masked_fill(t15, 0.0)
    return t16

def replacement_args(x):
    return (x,)

@triton.jit
def masked_distance_kernel(x_ptr, out_ptr, n, m):
    j = tl.program_id(0)
    k = tl.program_id(1)
    for l in range(m):
        x_jk = tl.load(x_ptr + j * m + k)
        x_jl = tl.load(x_ptr + j * m + l)
        is_diff = x_jk != x_jl
        out_val = tl.where(is_diff, -1000.0, 0.0)
        tl.store(out_ptr + j * m * m + k * m + l, out_val)

@torch.fx.wrap
def masked_distance(x):
    batch, n, m = x.shape
    out = torch.empty((batch, n, m, m), dtype=x.dtype, device=x.device)
    x_2d = x.view(n, m)
    masked_distance_kernel[(n, m)](x_2d, out, n, m)
    return out

def replacement_func():
    return masked_distance