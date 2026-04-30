import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_se_block_kernel(
    bias_ptr,
    weight_ptr,
    feature_ptr,
    input_conv_ptr,
    output_ptr,
    B, C_out, C_in, HW,
    w_s0, w_s1,
    f_s0, f_s1, f_s2, f_s3,
    i_s0, i_s1,
    b_s0,
    o_s0, o_s1,
    BLOCK_C: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_W: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_off < C_out

    # ---- Step 1: Conv2d (1x1) = matmul + bias ----
    bias_vals = tl.load(bias_ptr + c_off * b_s0, mask=c_mask, other=0.0).to(tl.float32)

    acc_conv = tl.zeros([BLOCK_C], dtype=tl.float32)
    for ci_start in range(0, C_in, BLOCK_CI):
        ci_off = ci_start + tl.arange(0, BLOCK_CI)
        ci_mask = ci_off < C_in

        input_vals = tl.load(
            input_conv_ptr + pid_b * i_s0 + ci_off * i_s1,
            mask=ci_mask, other=0.0
        ).to(tl.float32)

        weight_vals = tl.load(
            weight_ptr + c_off[:, None] * w_s0 + ci_off[None, :] * w_s1,
            mask=c_mask[:, None] & ci_mask[None, :], other=0.0
        ).to(tl.float32)

        acc_conv += tl.sum(weight_vals * input_vals[None, :], axis=1)

    conv_vals = acc_conv + bias_vals

    # ---- Step 2: Sigmoid ----
    sig_vals = 1.0 / (1.0 + tl.exp(-conv_vals))

    # ---- Steps 3-5: Multiply + GELU + AvgPool (row-based, unrolled) ----
    acc_spatial = tl.zeros([BLOCK_C], dtype=tl.float32)
    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    for h in range(H):
        feat_vals = tl.load(
            feature_ptr + pid_b * f_s0
            + c_off[:, None] * f_s1
            + h * f_s2
            + w_off[None, :] * f_s3,
            mask=c_mask[:, None] & w_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scaled = feat_vals * sig_vals[:, None]

        gelu_vals = scaled * 0.5 * (1.0 + tl.math.erf(scaled / 1.4142135623730951))

        valid = (c_mask[:, None] & w_mask[None, :]).to(tl.float32)
        acc_spatial += tl.sum(gelu_vals * valid, axis=1)

    avg_vals = acc_spatial / HW

    tl.store(output_ptr + pid_b * o_s0 + c_off * o_s1, avg_vals, mask=c_mask)


@torch.fx.wrap
def fused_se_block(bias, weight, feature, input_conv):
    B = feature.shape[0]
    C_out = feature.shape[1]
    C_in = input_conv.shape[1]
    H = feature.shape[2]
    W = feature.shape[3]
    HW = H * W

    BLOCK_C = 16
    BLOCK_CI = 64
    # Choose smallest power-of-2 BLOCK_W >= W to minimize waste
    BLOCK_W = 8 if W <= 8 else 16

    output = torch.empty((B, C_out), dtype=feature.dtype, device=feature.device)

    grid = (B, triton.cdiv(C_out, BLOCK_C))

    fused_se_block_kernel[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        feature_ptr=feature,
        input_conv_ptr=input_conv,
        output_ptr=output,
        B=B, C_out=C_out, C_in=C_in, HW=HW,
        w_s0=weight.stride(0), w_s1=weight.stride(1),
        f_s0=feature.stride(0), f_s1=feature.stride(1),
        f_s2=feature.stride(2), f_s3=feature.stride(3),
        i_s0=input_conv.stride(0), i_s1=input_conv.stride(1),
        b_s0=bias.stride(0),
        o_s0=output.stride(0), o_s1=output.stride(1),
        BLOCK_C=BLOCK_C,
        BLOCK_CI=BLOCK_CI,
        BLOCK_W=BLOCK_W,
        H=H,
        W=W,
    )

    return output


def replacement_func():
    return fused_se_block