import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_4


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit

def _mean_view_kernel(
    x_ptr,
    out_ptr,
    C,
    H,
    W,
    HW,
    stride_xc,
    stride_xh,
    stride_xw,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    if c < C:
        base = c * stride_xc
        acc = 0.0
        for start in tl.range(0, HW, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W
            w = offs - h * W
            ptrs = x_ptr + base + h * stride_xh + w * stride_xw
            x = tl.load(ptrs, mask=mask, other=0.0)
            acc = acc + tl.sum(x.to(tl.float32), axis=0)
        mean = acc / HW
        tl.store(out_ptr + c, mean)


@torch.fx.wrap
def triton_mean_view(x):
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    out = torch.empty((1, 1, c), device=x.device, dtype=x.dtype)
    grid = (c,)
    _mean_view_kernel[grid](
        x,
        out,
        c,
        h,
        w,
        hw,
        x.stride(1),
        x.stride(2),
        x.stride(3),
    )
    return out


def replacement_func():
    return triton_mean_view