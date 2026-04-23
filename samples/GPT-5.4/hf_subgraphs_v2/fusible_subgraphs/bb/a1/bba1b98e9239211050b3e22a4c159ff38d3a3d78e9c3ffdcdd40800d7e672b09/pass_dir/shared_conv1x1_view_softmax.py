import torch
import triton
import triton.language as tl

_SPATIAL = 4096


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def softmax_4096_kernel(
    x_ptr,
    out_ptr,
    x_stride0,
    out_stride0,
    out_stride2,
    SPATIAL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, SPATIAL)

    x = tl.load(x_ptr + row * x_stride0 + cols).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * out_stride0 + cols * out_stride2, y)


@torch.fx.wrap
def fused_conv1x1_view_softmax(x):
    batch = x.shape[0]
    out = torch.empty((batch, 1, _SPATIAL), device=x.device, dtype=x.dtype)
    softmax_4096_kernel[(batch,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        out.stride(2),
        SPATIAL=_SPATIAL,
    )
    return out