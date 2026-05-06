import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_mean1_kernel(
    in_ptr,
    out_ptr,
    mean_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for r in range(0, HW, BLOCK_SIZE):
        offsets = r + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        x = tl.load(in_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += x

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, sum_val.to(in_ptr.dtype.element_ty))
    tl.store(mean_ptr + pid, sum_val / HW)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_mean2_kernel(
    a_ptr, b_ptr, out_ptr, mean_ptr, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for r in range(0, HW, BLOCK_SIZE):
        offsets = r + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        a = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += a + b

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, sum_val.to(a_ptr.dtype.element_ty))
    tl.store(mean_ptr + pid, sum_val / HW)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _fused_add_mean3_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr, mean_ptr, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for r in range(0, HW, BLOCK_SIZE):
        offsets = r + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        a = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += a + b + c

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, sum_val.to(a_ptr.dtype.element_ty))
    tl.store(mean_ptr + pid, sum_val / HW)


@torch.fx.wrap
def _fused_add_mean_dispatch(a, b, c, route):
    N = a.shape[0]
    C = a.shape[1]
    H = a.shape[2]
    W = a.shape[3]
    HW = H * W

    out = torch.empty_like(a)
    mean_buf = torch.empty((N, C, 1, 1), dtype=a.dtype, device=a.device)
    grid = (N * C,)

    if route == "add0_mean":
        _fused_add_mean1_kernel[grid](a, out, mean_buf, HW)
    elif route == "add2_mean":
        _fused_add_mean2_kernel[grid](a, b, out, mean_buf, HW)
    elif route == "add3_mean":
        _fused_add_mean3_kernel[grid](a, b, c, out, mean_buf, HW)

    return out, mean_buf


def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1, in_1, "add2_mean")


def replacement_func():
    return _fused_add_mean_dispatch