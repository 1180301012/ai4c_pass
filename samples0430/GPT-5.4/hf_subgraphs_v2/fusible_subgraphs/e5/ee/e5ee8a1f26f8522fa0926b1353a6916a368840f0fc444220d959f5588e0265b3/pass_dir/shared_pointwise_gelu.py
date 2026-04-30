import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'W'],
)
@triton.jit
def _pointwise1x1_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    W,
    OC,
    sN,
    sIC,
    sH,
    sW,
    ws0,
    ws1,
    ysN,
    ysOC,
    ysH,
    ysW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < OC

    HW = sN // sH
    batch = offs_m // HW
    hw = offs_m - batch * HW
    h = hw // W
    w = hw - h * W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, 64, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < 64

        x_ptrs = x_ptr + batch[:, None] * sN + offs_k[None, :] * sIC + h[:, None] * sH + w[:, None] * sW
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = w_ptr + offs_k[:, None] * ws1 + offs_n[None, :] * ws0
        wv = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x, wv)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))

    y_ptrs = y_ptr + batch[:, None] * ysN + offs_n[None, :] * ysOC + h[:, None] * ysH + w[:, None] * ysW
    tl.store(y_ptrs, gelu, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_pointwise1x1_gelu(bias, weight, x):
    n, ic, h, w = x.shape
    oc = weight.shape[0]
    out = torch.empty((n, oc, h, w), device=x.device, dtype=x.dtype)
    m = n * h * w
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(oc, meta['BLOCK_N']))
    _pointwise1x1_gelu_kernel[grid](
        x,
        weight,
        bias,
        out,
        m,
        w,
        oc,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out