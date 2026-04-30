import torch
import triton
import triton.language as tl


@triton.jit
def _mean_hw_kernel_strided(
    x_ptr,
    out_ptr,
    c_dim,
    h,
    w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // c_dim
    c = pid - n * c_dim

    base = x_ptr + n * x_stride_n + c * x_stride_c
    hw = h * w
    acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, 16384, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < hw
        hh = idx // w
        ww = idx - hh * w
        vals = tl.load(base + hh * x_stride_h + ww * x_stride_w, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)

    acc = acc / hw
    tl.store(out_ptr + pid, acc)


@torch.fx.wrap
def triton_mean_hw_keepdim(x):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
    nc = n * c

    if h * w <= 1024:
        block = 256
        num_warps = 4
    elif h * w <= 4096:
        block = 512
        num_warps = 4
    else:
        block = 1024
        num_warps = 8

    _mean_hw_kernel_strided[(nc,)](
        x,
        out,
        c,
        h,
        w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK=block,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def pattern(x):
    tmp_2 = x.mean((2, 3), keepdim=True)
    return tmp_2


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_hw_keepdim