import torch
import triton
import triton.language as tl

def pattern(x):
    x_f32 = x.to(dtype=torch.float32)
    x_sq = x_f32 ** 2
    x_mean = x_sq.mean(dim=-1, keepdim=True)
    x_eps = x_mean + 1e-6
    x_rsqrt = torch.rsqrt(x_eps)
    return x_f32 * x_rsqrt

def replacement_args(x):
    return (x,)

def replacement_func():
    return layer_norm

@triton.jit
def layer_norm_kernel(x_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE, 1)
    mask = offsets < N

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x_vals * x_vals
    x_mean = tl.sum(x_sq, axis=0) / D
    x_eps = x_mean + 1e-6
    x_rsqrt = tl.rsqrt(x_eps)
    out_vals = x_vals * x_rsqrt
    tl.store(out_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def layer_norm(x):
    N = x.numel()
    D = x.size(-1)
    BLOCK_SIZE = 128
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        N=N,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out