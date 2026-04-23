import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_9):
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    tmp_5 = conv2d.view(1, 512, 64, 64)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_9):
    return (in_0, in_1, in_2, in_3, in_4, in_9)


@triton.jit
def depthwise_conv_bn_relu_kernel(
    conv_input_ptr,
    conv_weight_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    out_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    k_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid // out_h
    oh = pid % out_h

    # Load BN parameters for this channel
    bn_mean = tl.load(bn_mean_ptr + c).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + c).to(tl.float32)
    bn_gamma = tl.load(bn_weight_ptr + c).to(tl.float32)
    bn_beta = tl.load(bn_bias_ptr + c).to(tl.float32)

    # Precompute BN scale and offset: BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    # = (gamma / sqrt(var + eps)) * x + (beta - mean * gamma / sqrt(var + eps))
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    bn_scale = bn_gamma * inv_std
    bn_offset = bn_beta - bn_mean * bn_scale

    # Output column offsets
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < out_w

    # Convolution accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for kh in range(k_size):
        for kw in range(k_size):
            # Load weight element: weight[c, 0, kh, kw]
            w_off = c * k_size * k_size + kh * k_size + kw
            w = tl.load(conv_weight_ptr + w_off).to(tl.float32)

            # Load input elements for all output columns
            # input[c, oh+kh, ow+kw]
            ih = oh + kh
            iw = ow + kw
            i_off = c * in_h * in_w + ih * in_w + iw
            inp = tl.load(conv_input_ptr + i_off, mask=ow_mask, other=0.0).to(tl.float32)
            acc += inp * w

    # BN + ReLU
    result = acc * bn_scale + bn_offset
    result = tl.maximum(result, 0.0)

    # Store result: out[c, oh, ow]
    out_off = c * out_h * out_w + oh * out_w + ow
    tl.store(out_ptr + out_off, result, mask=ow_mask)


@torch.fx.wrap
def fused_conv_bn_relu(in_0, in_1, in_2, in_3, in_4, in_9):
    dtype = in_9.dtype
    device = in_9.device
    out = torch.empty(1, 512, 64, 64, dtype=dtype, device=device)

    BLOCK_W = 64
    total_programs = 512 * 64  # num_channels * out_h

    depthwise_conv_bn_relu_kernel[(total_programs,)](
        in_9, in_4, in_0, in_1, in_3, in_2, out,
        in_h=70, in_w=70, out_h=64, out_w=64,
        k_size=7, eps=1e-05,
        BLOCK_W=BLOCK_W,
    )

    return out


def replacement_func():
    return fused_conv_bn_relu