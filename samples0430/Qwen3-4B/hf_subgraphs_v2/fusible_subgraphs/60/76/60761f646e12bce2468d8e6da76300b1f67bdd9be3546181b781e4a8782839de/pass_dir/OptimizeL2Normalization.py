import torch
import triton
import triton.language as tl

def pattern(in_0):
    out = torch.cat([in_0], dim=1)
    out = torch.nn.functional.normalize(out, p=2, dim=1)
    return out

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start = row_id * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, C)
    row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row = tl.load(\
        x_ptr + start,
        mask=tl.arange(0, BLOCK_SIZE) < (C - start),
        other=0.0
    )
    sq = row * row
    sum_sq = tl.sum(sq)
    norm = tl.sqrt(sum_sq)
    tl.store(\
        out_ptr + start,
        row / norm,
        mask=tl.arange(0, BLOCK_SIZE) < (C - start)
    )

@torch.fx.wrap
def normalize_wrapper(x):
    N, C = x.shape
    out = torch.empty_like(x)
    normalize_kernel[ (N,) ](\
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        BLOCK_SIZE=256,
    )
    return out

def replacement_func():
    return normalize_wrapper