import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_2 = in_0 // 32
    tmp_3 = torch.sym_sum([1, tmp_2])
    tmp_4 = tmp_1.view(1, 1, -1)
    return (tmp_0, tmp_4)


def replacement_args(in_0, in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit

def _silu_mean_inplace_kernel(
    x_ptr,
    mean_ptr,
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
            x_f32 = x.to(tl.float32)
            y_f32 = x_f32 * tl.sigmoid(x_f32)
            tl.store(ptrs, y_f32, mask=mask)
            acc = acc + tl.sum(y_f32, axis=0)
        mean = acc / HW
        tl.store(mean_ptr + c, mean)


@torch.fx.wrap
def triton_silu_mean_view_inplace(x):
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    mean = torch.empty((1, 1, c), device=x.device, dtype=x.dtype)
    grid = (c,)
    _silu_mean_inplace_kernel[grid](
        x,
        mean,
        c,
        h,
        w,
        hw,
        x.stride(1),
        x.stride(2),
        x.stride(3),
    )
    return (x, mean)


def replacement_func():
    return triton_silu_mean_view_inplace