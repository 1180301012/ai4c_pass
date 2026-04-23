import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(a):
    tmp_2 = a.mean((2, 3), keepdim=True)
    return tmp_2


# Argument extraction function
def replacement_args(a):
    return (a,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def spatial_mean_keepdim_kernel(
    x_ptr,
    out_ptr,
    HW,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    n = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    x_base = x_ptr + n * stride_n + c * stride_c
    for start in range(0, HW, BLOCK_SIZE):
        idx = start + offs
        mask = idx < HW
        h = idx // W
        w = idx % W
        vals = tl.load(x_base + h * stride_h + w * stride_w, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) / HW
    out_ptr_nc = out_ptr + n * out_stride_n + c * out_stride_c
    tl.store(out_ptr_nc, mean_val, mask=(n < N) & (c < C))


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def spatial_mean_keepdim_triton(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (C, N)
    spatial_mean_keepdim_kernel[grid](
        x,
        out,
        H * W,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        N,
        C,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return spatial_mean_keepdim_triton