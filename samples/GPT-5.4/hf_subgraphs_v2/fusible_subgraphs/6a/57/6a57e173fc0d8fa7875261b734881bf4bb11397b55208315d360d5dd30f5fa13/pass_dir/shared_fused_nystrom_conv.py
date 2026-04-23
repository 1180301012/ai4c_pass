import torch
import triton
import triton.language as tl


@triton.jit
def _fused_depthwise_conv_add_permute_kernel(
    value_ptr,
    context_ptr,
    weight_ptr,
    out_ptr,
    B,
    C,
    W,
    D,
    K,
    PAD,
    OUT_W,
    stride_vb,
    stride_vc,
    stride_vw,
    stride_vd,
    stride_cb,
    stride_cc,
    stride_cw,
    stride_cd,
    stride_wc,
    stride_wk,
    stride_od0,
    stride_od1,
    stride_od2,
    stride_ow,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    bc = pid_bc
    b = bc // C
    c = bc % C

    mask_d = d_offsets < D
    mask_w = w_offsets < OUT_W

    acc = tl.zeros((BLOCK_W, BLOCK_D), dtype=tl.float32)

    for k in range(0, 65):
        in_w = w_offsets + k - PAD
        mask_in_w = (in_w >= 0) & (in_w < W)

        v_ptrs = (
            value_ptr
            + b * stride_vb
            + c * stride_vc
            + in_w[:, None] * stride_vw
            + d_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_in_w[:, None] & mask_d[None, :], other=0.0)

        w = tl.load(weight_ptr + c * stride_wc + k * stride_wk)
        acc += v.to(tl.float32) * w

    c_ptrs = (
        context_ptr
        + b * stride_cb
        + c * stride_cc
        + w_offsets[:, None] * stride_cw
        + d_offsets[None, :] * stride_cd
    )
    cval = tl.load(c_ptrs, mask=mask_w[:, None] & mask_d[None, :], other=0.0)
    acc += cval.to(tl.float32)

    out_ptrs = (
        out_ptr
        + b * stride_od0
        + w_offsets[:, None] * stride_od1
        + (c * D + d_offsets[None, :]) * stride_od2
    )

    out_dtype = cval.dtype
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_w[:, None] & mask_d[None, :])


@torch.fx.wrap
def fused_depthwise_conv_add_permute(value, context, weight, out_shape):
    B, C, W, D = value.shape
    K = weight.shape[2]
    PAD = 32
    OUT_W = W

    if tuple(out_shape) != (B, W, C * D):
        raise RuntimeError(f"unexpected out_shape {out_shape} for input shape {tuple(value.shape)}")

    out = torch.empty(out_shape, device=value.device, dtype=value.dtype)

    BLOCK_D = 32 if D >= 32 else 16 if D >= 16 else 8
    BLOCK_W = 32 if W >= 32 else 16

    grid = (
        triton.cdiv(D, BLOCK_D),
        triton.cdiv(OUT_W, BLOCK_W),
        B * C,
    )

    _fused_depthwise_conv_add_permute_kernel[grid](
        value,
        context,
        weight,
        out,
        B,
        C,
        W,
        D,
        K,
        PAD,
        OUT_W,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        context.stride(0),
        context.stride(1),
        context.stride(2),
        context.stride(3),
        weight.stride(0),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        0,
        BLOCK_D=BLOCK_D,
        BLOCK_W=BLOCK_W,
    )
    return out


@torch.fx.wrap
def shared_replacement(value, context, weight, out_shape):
    return fused_depthwise_conv_add_permute(value, context, weight, out_shape)