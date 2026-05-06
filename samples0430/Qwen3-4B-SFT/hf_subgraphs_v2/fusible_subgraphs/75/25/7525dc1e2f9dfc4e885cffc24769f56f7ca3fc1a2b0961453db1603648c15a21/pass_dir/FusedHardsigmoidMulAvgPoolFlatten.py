import torch
import triton
import triton.language as tl


@triton.jit
def fused_hs_mp_kernel(
    in2_ptr,
    conv_ptr,
    out_ptr,
    C,
    HW,
    stride_b_in2,
    stride_c_in2,
    stride_b_conv,
    stride_c_conv,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Load conv output for this (b, c) — shape [B, C, 1, 1], stride for last two dims = 1
    conv_val = tl.load(conv_ptr + b * stride_b_conv + c * stride_c_conv)

    # hardsigmoid: clamp(x/6 + 0.5, 0, 1)  — computed in float32 for numerical stability
    conv_f32 = conv_val.to(tl.float32)
    tmp_f32 = conv_f32 / 6.0 + 0.5
    tmp_f32 = tl.maximum(tmp_f32, 0.0)
    tmp_f32 = tl.minimum(tmp_f32, 1.0)

    # Dtype tag for output cast
    in2_f32 = tl.load(in2_ptr + b * stride_b_in2 + c * stride_c_in2).to(tl.float32)
    xi_f32 = tmp_f32 * in2_f32

    # Reduction over H*W using a tiled loop
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    mask = tl.arange(0, BLOCK_HW) < HW

    for i in range(0, HW, BLOCK_HW):
        hw_offs = i + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        x = tl.load(
            in2_ptr + b * stride_b_in2 + c * stride_c_in2 + hw_offs,
            mask=hw_mask, other=0.0
        ).to(tl.float32)
        acc += tl.where(hw_mask, xi_f32 * x, 0.0)

    total = tl.sum(acc)
    out_f32 = total / HW

    # Convert output back to the original dtype (orig dtype of in_2)
    tl.store(out_ptr + b * C + c, out_f32.to(in2_f32.dtype))


@torch.fx.wrap
def fused_hs_mp(in2, conv_out):
    B = in2.shape[0]
    C = in2.shape[1]
    H = in2.shape[2]
    W = in2.shape[3]
    HW = H * W

    # Choose the smallest power-of-2 BLOCK_HW >= HW (no warp divergence from masking)
    if HW <= 64:
        BLOCK_HW = 64
    elif HW <= 128:
        BLOCK_HW = 128
    elif HW <= 256:
        BLOCK_HW = 256
    elif HW <= 512:
        BLOCK_HW = 512
    else:
        BLOCK_HW = 1024

    out = torch.empty((B, C), dtype=in2.dtype, device=in2.device)

    grid = (B * C,)
    fused_hs_mp_kernel[grid](
        in2, conv_out, out,
        C, HW,
        in2.stride(0), in2.stride(1),
        conv_out.stride(0), conv_out.stride(1),
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return (out,)


# ---------------------------------------------------------------------------
# Pattern: hardsigmoid(conv_out) * in2  →  avg_pool2d=1  →  flatten  →  dropout(p=0)
# This exactly mirrors what model.py computes for ops 2-6.
# ---------------------------------------------------------------------------
def pattern(in_2, conv_out):
    tmp_3 = torch.nn.functional.hardsigmoid(conv_out, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return (tmp_7,)


def replacement_args(in_2, conv_out):
    return (in_2, conv_out)


def replacement_func():
    return fused_hs_mp