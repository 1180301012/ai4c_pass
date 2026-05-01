import torch
import triton
import triton.language as tl

def pattern(x):
    t = x.norm(p=2, dim=-1, keepdim=True)
    return x / t

def replacement_args(x):
    return (x, )

@triton.jit
def normalize_kernel(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * N
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq)
    norm = tl.sqrt(sum_sq)
    inv_norm = 1.0 / norm
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        out = x * inv_norm
        tl.store(out_ptr + row_start + cols, out, mask=mask)

@torch.fx.wrap
def normalized(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (M, 1)
    normalize_kernel[grid](x, out, M, N, BLOCK_SIZE)
    return out

def replacement_func():
    return normalized