import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _l2_normalize_dim1_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(sum_sq)
    inv_norm = 1.0 / tl.maximum(norm, eps)
    y = x_f32 * inv_norm

    tl.store(out_ptr + row * stride_om + offs * stride_on, y, mask=mask)


@torch.fx.wrap
def triton_l2_normalize_dim1(x):
    m = x.shape[0]
    n = x.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (m,)
    _l2_normalize_dim1_kernel[grid](
        x,
        out,
        n,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        1e-12,
    )
    return out