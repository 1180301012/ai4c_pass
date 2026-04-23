import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_batchnorm_relu_kernel(
    input_ptr, weight_ptr, bias_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    batch_size, in_channels, height, width,
    out_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * height * width
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < n_elements
    
    tmp = output_idx
    w = tmp % width
    tmp = tmp // width
    h = tmp % height
    tmp = tmp // height
    ch = tmp % out_channels
    batch = tmp // out_channels
    
    # Conv2d: 1x1 kernel, stride 1, no padding
    conv_out = tl.load(bias_ptr + ch).to(tl.float32)
    
    for c_in in range(in_channels):
        input_idx = ((batch * in_channels + c_in) * height + h) * width + w
        inp = tl.load(input_ptr + input_idx, mask=mask, other=0.0).to(tl.float32)
        weight_idx = ch * in_channels + c_in
        w_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
        conv_out += inp * w_val
    
    # BatchNorm
    mean = tl.load(bn_mean_ptr + ch).to(tl.float32)
    var = tl.load(bn_var_ptr + ch).to(tl.float32)
    bn_w = tl.load(bn_weight_ptr + ch).to(tl.float32)
    bn_b = tl.load(bn_bias_ptr + ch).to(tl.float32)
    
    var_eps = var + eps
    std = tl.sqrt(var_eps)
    bn_out = (conv_out - mean) / std * bn_w + bn_b
    
    # ReLU
    relu_out = tl.maximum(bn_out, 0.0)
    
    tl.store(output_ptr + output_idx, relu_out.to(tl.float16), mask=mask)


@torch.fx.wrap
def conv2d_batchnorm_relu_wrapper(input_tensor, conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias):
    B, C_in, H, W = input_tensor.shape
    C_out = conv_weight.shape[0]
    
    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = B * C_out * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv2d_batchnorm_relu_kernel[(num_programs,)](
        input_tensor, conv_weight, conv_bias,
        bn_mean, bn_var, bn_weight, bn_bias,
        output,
        B, C_in, H, W,
        C_out,
        1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_9, in_6):
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    return conv2d_1


def replacement_args(in_9, in_6):
    return (in_9, in_6)


def replacement_func():
    return conv2d_batchnorm_relu_wrapper