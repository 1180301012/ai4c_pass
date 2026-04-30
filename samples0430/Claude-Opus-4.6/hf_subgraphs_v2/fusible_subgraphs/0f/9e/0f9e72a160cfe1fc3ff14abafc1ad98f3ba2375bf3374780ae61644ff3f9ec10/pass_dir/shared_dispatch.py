import torch
import triton
import triton.language as tl


@triton.jit
def l2_normalize_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    norm_val = tl.sqrt(norm_sq)
    result = x / norm_val
    tl.store(out_ptr + offsets, result)


@triton.jit
def exp_mul_kernel(scalar_ptr, vector_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    s = tl.load(scalar_ptr).to(tl.float32)
    v = tl.load(vector_ptr + offsets).to(tl.float32)
    result = tl.exp(s) * v
    tl.store(out_ptr + offsets, result)


def dispatch(*args):
    route = args[0]
    if route == "norm_div":
        x = args[1]
        N = x.numel()
        out = torch.empty_like(x)
        l2_normalize_kernel[(1,)](x, out, N=N)
        return out
    elif route == "exp_mul":
        scalar = args[1]
        vector = args[2]
        N = vector.numel()
        out = torch.empty_like(vector)
        exp_mul_kernel[(1,)](scalar, vector, out, N=N)
        return out