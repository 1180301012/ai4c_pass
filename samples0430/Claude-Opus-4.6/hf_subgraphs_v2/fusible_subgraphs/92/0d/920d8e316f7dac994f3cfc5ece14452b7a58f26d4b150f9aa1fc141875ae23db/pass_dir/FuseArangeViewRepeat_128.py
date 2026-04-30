import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x,)


@triton.jit
def repeat_kernel_128(in_ptr, out_ptr):
    offsets = tl.arange(0, 128)
    vals = tl.load(in_ptr + offsets)
    tl.store(out_ptr + offsets, vals)
    tl.store(out_ptr + 128 + offsets, vals)


@triton.jit
def repeat_kernel_1024(in_ptr, out_ptr, N: tl.constexpr):
    offsets = tl.arange(0, 1024)
    mask = offsets < N
    vals = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)
    tl.store(out_ptr + N + offsets, vals, mask=mask)


@torch.fx.wrap
def view_repeat_optimized(x):
    N = x.numel()
    out = torch.empty((2, N), dtype=x.dtype, device=x.device)
    if N <= 128:
        repeat_kernel_128[(1,)](x, out, num_warps=1)
    else:
        repeat_kernel_1024[(1,)](x, out, N, num_warps=1)
    return out


def replacement_func():
    return view_repeat_optimized