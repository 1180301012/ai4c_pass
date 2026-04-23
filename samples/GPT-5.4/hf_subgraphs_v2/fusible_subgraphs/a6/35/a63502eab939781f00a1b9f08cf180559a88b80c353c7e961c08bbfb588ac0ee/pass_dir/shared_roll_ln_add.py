import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def _roll_layernorm_add_kernel(
    in3_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    H,
    W,
    C,
    INNER,
    SHIFT_H,
    SHIFT_W,
    eps,
    stride_in3_d1,
    stride_in3_d2,
    stride_in3_d3,
    stride_in3_d4,
    stride_in3_d5,
    stride_res_d1,
    stride_res_d2,
    stride_out_d1,
    stride_out_d2,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    hw = H * W
    if pid >= hw:
        return

    h = pid // W
    w = pid % W
    src_h = (h - SHIFT_H + H) % H
    src_w = (w - SHIFT_W + W) % W

    d1 = src_h // INNER
    d2 = src_h % INNER
    d3 = src_w // INNER
    d4 = src_w % INNER

    offs = tl.arange(0, BLOCK_N)
    mask = offs < C

    in3_base = d1 * stride_in3_d1 + d2 * stride_in3_d2 + d3 * stride_in3_d3 + d4 * stride_in3_d4
    x = tl.load(in3_ptr + in3_base + offs * stride_in3_d5, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / C
    centered = x_fp32 - mean
    centered = tl.where(mask, centered, 0.0)
    var = tl.sum(centered * centered, axis=0) / C
    inv_std = tl.rsqrt(var + eps)

    wv = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bv = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std
    y = y * wv + bv

    r = tl.load(residual_ptr + pid * stride_res_d1 + offs * stride_res_d2, mask=mask, other=0.0).to(tl.float32)
    out = (y + r).to(x.dtype)
    tl.store(out_ptr + pid * stride_out_d1 + offs * stride_out_d2, out, mask=mask)


@torch.fx.wrap
def fused_roll_layernorm_add_dispatch(in_0, in_1, in_2, in_3, route):
    if route == 's32_c768':
        H = 32
        W = 32
        C = 768
        INNER = 8
        SHIFT_H = 4
        SHIFT_W = 4
    elif route == 's64_c384':
        H = 64
        W = 64
        C = 384
        INNER = 8
        SHIFT_H = 4
        SHIFT_W = 4
    else:
        raise RuntimeError(f'Unknown route: {route}')

    out = torch.empty_like(in_2)
    grid = (H * W,)

    s_in3 = in_3.stride()
    s_res = in_2.stride()
    s_out = out.stride()

    _roll_layernorm_add_kernel[grid](
        in_3,
        in_2,
        in_1,
        in_0,
        out,
        H,
        W,
        C,
        INNER,
        SHIFT_H,
        SHIFT_W,
        1e-5,
        s_in3[1],
        s_in3[2],
        s_in3[3],
        s_in3[4],
        s_in3[5],
        s_res[1],
        s_res[2],
        s_out[1],
        s_out[2],
    )
    return out