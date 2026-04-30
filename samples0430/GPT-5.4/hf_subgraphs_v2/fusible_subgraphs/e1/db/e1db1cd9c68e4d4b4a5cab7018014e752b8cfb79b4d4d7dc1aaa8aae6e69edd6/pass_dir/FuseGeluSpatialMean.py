import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW', 'DTYPE_KIND'],
)
@triton.jit

def fused_gelu_mean_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    NC,
    HW,
    DTYPE_KIND,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= NC:
        return

    base = pid * HW
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        cdf = 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865475))
        gelu_f32 = x_f32 * cdf

        if DTYPE_KIND == 0:
            y = gelu_f32.to(tl.float16)
        elif DTYPE_KIND == 1:
            y = gelu_f32.to(tl.bfloat16)
        else:
            y = gelu_f32

        tl.store(y_ptr + base + offs, y, mask=mask)
        acc += tl.where(mask, gelu_f32, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / HW
    tl.store(mean_ptr + pid, mean)


@torch.fx.wrap
def fused_gelu_spatial_mean(x):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    nc = n * c

    y = torch.empty_like(x)
    mean = torch.empty((n, c, 1, 1), device=x.device, dtype=torch.float32)

    if x.dtype == torch.float16:
        dtype_kind = 0
    elif x.dtype == torch.bfloat16:
        dtype_kind = 1
    else:
        dtype_kind = 2

    grid = (nc,)
    fused_gelu_mean_kernel[grid](
        x,
        y,
        mean,
        nc,
        hw,
        dtype_kind,
    )
    return (y, mean)


def replacement_func():
    return fused_gelu_spatial_mean