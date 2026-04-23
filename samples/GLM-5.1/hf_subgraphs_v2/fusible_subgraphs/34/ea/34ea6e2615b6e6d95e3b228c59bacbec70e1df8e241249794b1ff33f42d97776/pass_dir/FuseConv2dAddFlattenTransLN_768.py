import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)
    tmp_5 = conv2d + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05)
    tmp_9 = tmp_8.transpose(0, 1)
    tmp_10 = tmp_8.transpose(0, 1)
    return (tmp_7, tmp_10, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_3, in_2, in_1, in_0, "route_768")

@triton.jit
def fused_conv_add_ln_kernel(
    input_ptr, weight_ptr, conv_bias_ptr, ln_weight_ptr, ln_bias_ptr,
    out7_ptr, out9_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    s_in_c, s_in_h, s_in_w,
    s_w_c, s_w_dh, s_w_dw,
    s_o7_p, s_o7_c,
    s_o9_p, s_o9_c,
    BLOCK_C: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    p = tl.program_id(0)
    h = p // W
    w = p % W

    c_offsets = tl.arange(0, BLOCK_C)

    total_sum = 0.0
    total_sum_sq = 0.0

    # Pass 1: depthwise conv + residual add + store to out7 + accumulate stats
    for c_block in range(NUM_BLOCKS):
        c_start = c_block * BLOCK_C
        c_off = c_start + c_offsets

        # Load center pixel for residual connection
        center = tl.load(input_ptr + c_off * s_in_c + h * s_in_h + w * s_in_w).to(tl.float32)

        # Depthwise conv: 3x3 kernel with padding=1
        conv_val = tl.zeros([BLOCK_C], dtype=tl.float32)
        conv_b = tl.load(conv_bias_ptr + c_off).to(tl.float32)

        for dh in range(3):
            for dw in range(3):
                ih = h + dh - 1
                iw = w + dw - 1
                ih_safe = tl.where(ih < 0, 0, tl.where(ih >= H, H - 1, ih))
                iw_safe = tl.where(iw < 0, 0, tl.where(iw >= W, W - 1, iw))
                is_valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                in_val = tl.load(input_ptr + c_off * s_in_c + ih_safe * s_in_h + iw_safe * s_in_w).to(tl.float32)
                w_val = tl.load(weight_ptr + c_off * s_w_c + dh * s_w_dh + dw * s_w_dw).to(tl.float32)
                conv_val += tl.where(is_valid, in_val * w_val, 0.0)

        conv_val += conv_b
        res_val = conv_val + center

        # Store to out7 [1, H*W, C]
        tl.store(out7_ptr + p * s_o7_p + c_off * s_o7_c, res_val)

        # Accumulate layer norm statistics
        total_sum += tl.sum(res_val)
        total_sum_sq += tl.sum(res_val * res_val)

    # Compute mean and variance
    mean_val = total_sum / C
    var_val = total_sum_sq / C - mean_val * mean_val
    rstd = 1.0 / tl.sqrt(var_val + 1e-5)

    # Pass 2: layer norm + store to out9
    for c_block in range(NUM_BLOCKS):
        c_start = c_block * BLOCK_C
        c_off = c_start + c_offsets

        res_val = tl.load(out7_ptr + p * s_o7_p + c_off * s_o7_c).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + c_off).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + c_off).to(tl.float32)

        ln_val = (res_val - mean_val) * rstd * ln_w + ln_b

        tl.store(out9_ptr + p * s_o9_p + c_off * s_o9_c, ln_val)

@torch.fx.wrap
def dispatch_wrapper(input, weight, conv_bias, ln_weight, ln_bias, route):
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    N_HW = H * W

    BLOCK_C = 128
    NUM_BLOCKS = C // BLOCK_C

    out7 = torch.empty(1, N_HW, C, dtype=input.dtype, device=input.device)
    out9 = torch.empty(N_HW, 1, C, dtype=input.dtype, device=input.device)

    grid = (N_HW,)

    fused_conv_add_ln_kernel[grid](
        input_ptr=input, weight_ptr=weight, conv_bias_ptr=conv_bias,
        ln_weight_ptr=ln_weight, ln_bias_ptr=ln_bias,
        out7_ptr=out7, out9_ptr=out9,
        C=C, H=H, W=W,
        s_in_c=input.stride(1), s_in_h=input.stride(2), s_in_w=input.stride(3),
        s_w_c=weight.stride(0), s_w_dh=weight.stride(2), s_w_dw=weight.stride(3),
        s_o7_p=out7.stride(1), s_o7_c=out7.stride(2),
        s_o9_p=out9.stride(0), s_o9_c=out9.stride(2),
        BLOCK_C=BLOCK_C, NUM_BLOCKS=NUM_BLOCKS,
    )

    return out7, out9, out9

def replacement_func():
    return dispatch_wrapper