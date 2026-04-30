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
def fused_conv_bn_relu_kernel(
    input_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    bn_w_ptr,
    bn_b_ptr,
    output_ptr,
    in_c_stride,
    in_h_stride,
    in_w_stride,
    wt_c_stride,
    wt_h_stride,
    wt_w_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    K_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)

    w_out = tl.arange(0, BLOCK_W)
    w_mask = w_out < OUT_W

    # Load BN parameters for this channel
    mean_val = tl.load(mean_ptr + pid_c).to(tl.float32)
    var_val = tl.load(var_ptr + pid_c).to(tl.float32)
    bn_w_val = tl.load(bn_w_ptr + pid_c).to(tl.float32)
    bn_b_val = tl.load(bn_b_ptr + pid_c).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var_val + EPS)

    # Base pointers for this channel
    in_base = input_ptr + pid_c * in_c_stride
    wt_base = weight_ptr + pid_c * wt_c_stride
    out_base = output_ptr + pid_c * out_c_stride

    # Compute depthwise conv
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for kh in range(K_SIZE):
        for kw in range(K_SIZE):
            # Load weight value (scalar, same for all spatial positions in this channel)
            wt_val = tl.load(wt_base + kh * wt_h_stride + kw * wt_w_stride).to(tl.float32)

            # Load input values (vector, different offsets for each output column)
            iw = w_out + kw
            ih = pid_h + kh
            in_val = tl.load(in_base + ih * in_h_stride + iw * in_w_stride, mask=w_mask, other=0.0).to(tl.float32)

            acc += in_val * wt_val

    # Apply batch norm: (acc - mean) * inv_std * bn_weight + bn_bias
    acc = (acc - mean_val) * inv_std * bn_w_val + bn_b_val

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store output
    tl.store(out_base + pid_h * out_h_stride + w_out * out_w_stride, acc, mask=w_mask)


@torch.fx.wrap
def fused_conv_bn_relu(running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_input):
    C = conv_input.shape[1]
    IN_H = conv_input.shape[2]
    IN_W = conv_input.shape[3]
    K = conv_weight.shape[2]
    OUT_H = IN_H - K + 1
    OUT_W = IN_W - K + 1

    output = torch.empty((1, C, OUT_H, OUT_W), dtype=conv_input.dtype, device=conv_input.device)

    grid = (C, OUT_H)

    fused_conv_bn_relu_kernel[grid](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        mean_ptr=running_mean,
        var_ptr=running_var,
        bn_w_ptr=bn_weight,
        bn_b_ptr=bn_bias,
        output_ptr=output,
        in_c_stride=conv_input.stride()[1],
        in_h_stride=conv_input.stride()[2],
        in_w_stride=conv_input.stride()[3],
        wt_c_stride=conv_weight.stride()[0],
        wt_h_stride=conv_weight.stride()[2],
        wt_w_stride=conv_weight.stride()[3],
        out_c_stride=output.stride()[1],
        out_h_stride=output.stride()[2],
        out_w_stride=output.stride()[3],
        IN_H=IN_H,
        IN_W=IN_W,
        OUT_H=OUT_H,
        OUT_W=OUT_W,
        K_SIZE=K,
        EPS=1e-5,
        BLOCK_W=OUT_W,
    )

    return output


def replacement_func():
    return fused_conv_bn_relu