import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv_ln_relu_kernel_19(
    input_ptr, weight_ptr, conv_bias_ptr,
    ln_weight_ptr, ln_bias_ptr, output_ptr,
    C_in, C_out,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    b = tl.program_id(0)

    c_out_range = tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_range < C_out

    # Compute matmul: output[c_out] = sum_c_in(input[c_in] * weight[c_out, c_in]) + bias[c_out]
    acc = tl.zeros([BLOCK_C_OUT], dtype=tl.float32)

    for c_in_start in range(0, C_in, BLOCK_C_IN):
        c_in_range = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_range < C_in

        # Load input vector for this batch element
        inp = tl.load(input_ptr + b * C_in + c_in_range, mask=c_in_mask, other=0.0).to(tl.float32)

        # Load weight tile [BLOCK_C_OUT, BLOCK_C_IN]
        w_offsets = c_out_range[:, None] * C_in + c_in_range[None, :]
        w_mask = c_out_mask[:, None] & c_in_mask[None, :]
        w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate partial dot products
        acc += tl.sum(w * inp[None, :], axis=1)

    # Add conv bias
    bias = tl.load(conv_bias_ptr + c_out_range, mask=c_out_mask, other=0.0).to(tl.float32)
    conv_out = acc + bias

    # Layer Norm: normalize over C_out dimension
    sum_val = tl.sum(tl.where(c_out_mask, conv_out, 0.0))
    mean = sum_val / C_out

    diff = conv_out - mean
    sum_sq = tl.sum(tl.where(c_out_mask, diff * diff, 0.0))
    var = sum_sq / C_out

    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    normalized = diff * inv_std

    # Affine transform
    ln_w = tl.load(ln_weight_ptr + c_out_range, mask=c_out_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + c_out_range, mask=c_out_mask, other=0.0).to(tl.float32)
    result = normalized * ln_w + ln_b

    # ReLU
    result = tl.maximum(result, 0.0)

    # Store output
    tl.store(output_ptr + b * C_out + c_out_range, result, mask=c_out_mask)


@torch.fx.wrap
def fused_conv_ln_relu_19(in_0, in_1, in_2, in_3, in_4):
    """
    Fused Conv2d(1x1) + LayerNorm + ReLU
    in_0: conv bias [C_out]
    in_1: conv weight [C_out, C_in, 1, 1]
    in_2: ln_bias [C_out, 1, 1]
    in_3: ln_weight [C_out, 1, 1]
    in_4: input [B, C_in, 1, 1]
    """
    B = in_4.shape[0]
    C_in = in_4.shape[1]
    C_out = in_1.shape[0]

    output = torch.empty((B, C_out, 1, 1), dtype=in_4.dtype, device=in_4.device)

    BLOCK_C_OUT = triton.next_power_of_2(C_out)
    BLOCK_C_IN = 64

    grid = (B,)
    _fused_conv_ln_relu_kernel_19[grid](
        in_4, in_1, in_0,
        in_3, in_2, output,
        C_in, C_out,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
    )

    return output


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (19, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_conv_ln_relu_19