import torch
import triton
import triton.language as tl


def pattern(a, b, c):
    t = a * c
    t0 = t.float()
    t1 = t0.pow(2)
    t2 = t1.mean(-1, keepdim=True)
    t3 = t2 + 1e-06
    t4 = torch.rsqrt(t3)
    t5 = t0 * t4
    t6 = b.float()
    t7 = 1.0 + t6
    t8 = t5 * t7
    return t, t8

def replacement_args(a, b, c):
    return (a, b, c)


@triton.jit
def norm_kernel(x_ptr, out_ptr, B, S, H, eps, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    sum_sq = tl.zeros([], dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        x = tl.load(x_ptr + (b * S * H + s * H + i), 
                    mask=(i + tl.arange(0, BLOCK_SIZE)) < H, 
                    other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq)
    mean_sq = sum_sq / tl.float32(H)
    inv_std = tl.rsqrt(mean_sq + eps)
    for i in range(0, H, BLOCK_SIZE):
        x = tl.load(x_ptr + (b * S * H + s * H + i), 
                    mask=(i + tl.arange(0, BLOCK_SIZE)) < H, 
                    other=0.0)
        out = x * inv_std
        tl.store(out_ptr + (b * S * H + s * H + i), 
                 out, 
                 mask=(i + tl.arange(0, BLOCK_SIZE)) < H)


@torch.fx.wrap
def norm_kernel_wrapper(a, b, c):
    t = a * c
    t0 = t.float()
    B, S, H = t0.shape
    out = torch.empty_like(t0)
    norm_kernel[(B, S)](
        t0, out, B, S, H, 1e-6, BLOCK_SIZE=256
    )
    b_float = b.float()
    b_factor = 1.0 + b_float
    result = out * b_factor
    return t, result

def replacement_func():
    return norm_kernel_wrapper