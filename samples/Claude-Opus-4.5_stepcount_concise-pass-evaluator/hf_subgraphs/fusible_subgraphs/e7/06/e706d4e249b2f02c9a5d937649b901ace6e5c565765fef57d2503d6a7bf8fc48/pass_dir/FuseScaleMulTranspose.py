import torch
import triton
import triton.language as tl

def pattern(in_1, scale):
    tmp_0 = in_1 * scale
    return tmp_0

def replacement_args(in_1, scale):
    return (in_1, scale)

@triton.jit
def scale_kernel(x_ptr, out_ptr, scale, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scale, mask=mask)

@torch.fx.wrap
def scale_mul(x, s):
    N = x.numel()
    out = torch.empty_like(x)
    grid = (N + 2047) // 2048
    scale_kernel[(grid,)](x, out, s, N, BLOCK=2048, num_warps=8)
    return out

def replacement_func():
    return scale_mul