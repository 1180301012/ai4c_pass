import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _depthwise3x3_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    C,
    H,
    W,
    HW,
    sN,
    sC,
    sH,
    sW,
    ws0,
    ws2,
    ws3,
    ysN,
    ysC,
    ysH,
    ysW,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW
    h = offs // W
    w = offs % W

    b = tl.load(b_ptr + c).to(tl.float32)
    w00 = tl.load(w_ptr + c * ws0 + 0 * ws2 + 0 * ws3).to(tl.float32)
    w01 = tl.load(w_ptr + c * ws0 + 0 * ws2 + 1 * ws3).to(tl.float32)
    w02 = tl.load(w_ptr + c * ws0 + 0 * ws2 + 2 * ws3).to(tl.float32)
    w10 = tl.load(w_ptr + c * ws0 + 1 * ws2 + 0 * ws3).to(tl.float32)
    w11 = tl.load(w_ptr + c * ws0 + 1 * ws2 + 1 * ws3).to(tl.float32)
    w12 = tl.load(w_ptr + c * ws0 + 1 * ws2 + 2 * ws3).to(tl.float32)
    w20 = tl.load(w_ptr + c * ws0 + 2 * ws2 + 0 * ws3).to(tl.float32)
    w21 = tl.load(w_ptr + c * ws0 + 2 * ws2 + 1 * ws3).to(tl.float32)
    w22 = tl.load(w_ptr + c * ws0 + 2 * ws2 + 2 * ws3).to(tl.float32)

    hm1 = h - 1
    hp1 = h + 1
    wm1 = w - 1
    wp1 = w + 1

    hm1_ok = hm1 >= 0
    hp1_ok = hp1 < H
    wm1_ok = wm1 >= 0
    wp1_ok = wp1 < W

    hm1s = tl.where(hm1_ok, hm1, 0)
    hp1s = tl.where(hp1_ok, hp1, 0)
    wm1s = tl.where(wm1_ok, wm1, 0)
    wp1s = tl.where(wp1_ok, wp1, 0)

    base = x_ptr + n * sN + c * sC
    acc = tl.full((BLOCK_HW,), b, tl.float32)
    acc += tl.load(base + hm1s * sH + wm1s * sW, mask=mask & hm1_ok & wm1_ok, other=0.0).to(tl.float32) * w00
    acc += tl.load(base + hm1s * sH + w * sW,    mask=mask & hm1_ok,          other=0.0).to(tl.float32) * w01
    acc += tl.load(base + hm1s * sH + wp1s * sW, mask=mask & hm1_ok & wp1_ok, other=0.0).to(tl.float32) * w02
    acc += tl.load(base + h * sH + wm1s * sW,    mask=mask & wm1_ok,           other=0.0).to(tl.float32) * w10
    acc += tl.load(base + h * sH + w * sW,       mask=mask,                     other=0.0).to(tl.float32) * w11
    acc += tl.load(base + h * sH + wp1s * sW,    mask=mask & wp1_ok,            other=0.0).to(tl.float32) * w12
    acc += tl.load(base + hp1s * sH + wm1s * sW, mask=mask & hp1_ok & wm1_ok, other=0.0).to(tl.float32) * w20
    acc += tl.load(base + hp1s * sH + w * sW,    mask=mask & hp1_ok,          other=0.0).to(tl.float32) * w21
    acc += tl.load(base + hp1s * sH + wp1s * sW, mask=mask & hp1_ok & wp1_ok, other=0.0).to(tl.float32) * w22

    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))
    tl.store(y_ptr + n * ysN + c * ysC + h * ysH + w * ysW, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'OC'],
)
@triton.jit
def _pointwise1x1_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    W,
    HW,
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

    n = offs_m // HW
    hw = offs_m % HW
    h = hw // W
    w = hw % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, 64, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < 64

        x_ptrs = x_ptr + n[:, None] * sN + offs_k[None, :] * sIC + h[:, None] * sH + w[:, None] * sW
        xv = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        w_ptrs = w_ptr + offs_k[:, None] * ws1 + offs_n[None, :] * ws0
        wv = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(xv, wv)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))

    y_ptrs = y_ptr + n[:, None] * ysN + offs_n[None, :] * ysOC + h[:, None] * ysH + w[:, None] * ysW
    tl.store(y_ptrs, y.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def _run_depthwise3x3(x, w, b):
    N, C, H, W = x.shape
    out = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(H * W, meta['BLOCK_HW']), N * C)
    _depthwise3x3_gelu_kernel[grid](
        x, w, b, out,
        C, H, W, H * W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


def _run_pointwise1x1(x, w, b):
    N, IC, H, W = x.shape
    OC = w.shape[0]
    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    M = N * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))
    _pointwise1x1_gelu_kernel[grid](
        x, w, b, out,
        M, W, HW, OC,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


@torch.fx.wrap
def fused_conv_gelu_dispatch(bias, weight, x, route):
    if route == 'depthwise3x3':
        return _run_depthwise3x3(x, weight, bias)
    if route == 'pointwise1x1':
        return _run_pointwise1x1(x, weight, bias)
    return torch.empty((0,), device=x.device, dtype=x.dtype)