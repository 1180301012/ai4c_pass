import torch
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv1d_kernel(
    value_ptr,
    weight_ptr,
    out_ptr,
    B,
    C,
    W,
    D,
    PAD,
    stride_vb,
    stride_vc,
    stride_vw,
    stride_vd,
    stride_wc,
    stride_wk,
    stride_ob,
    stride_oc,
    stride_ow,
    stride_od,
    BLOCK_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    b = pid_bc // C
    c = pid_bc % C

    mask_d = d_offsets < D
    mask_w = w_offsets < W

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

    out_ptrs = (
        out_ptr
        + b * stride_ob
        + c * stride_oc
        + w_offsets[:, None] * stride_ow
        + d_offsets[None, :] * stride_od
    )
    out_elem = tl.load(out_ptrs, mask=mask_w[:, None] & mask_d[None, :], other=0.0)
    tl.store(out_ptrs, acc.to(out_elem.dtype), mask=mask_w[:, None] & mask_d[None, :])


@torch.fx.wrap
def shared_depthwise_conv_replacement(value, weight):
    B, C, W, D = value.shape
    out = torch.empty((B, C, W, D), device=value.device, dtype=value.dtype)

    BLOCK_W = 32 if W >= 32 else 16
    BLOCK_D = 32 if D >= 32 else 16 if D >= 16 else 8

    grid = (
        triton.cdiv(D, BLOCK_D),
        triton.cdiv(W, BLOCK_W),
        B * C,
    )

    _depthwise_conv1d_kernel[grid](
        value,
        weight,
        out,
        B,
        C,
        W,
        D,
        32,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        weight.stride(0),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_W=BLOCK_W,
        BLOCK_D=BLOCK_D,
    )
    return out