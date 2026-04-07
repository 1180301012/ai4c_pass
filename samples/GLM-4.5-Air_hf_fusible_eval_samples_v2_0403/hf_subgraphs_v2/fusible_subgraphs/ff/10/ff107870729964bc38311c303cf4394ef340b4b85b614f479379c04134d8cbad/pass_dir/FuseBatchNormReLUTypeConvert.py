import torch
import triton
import triton.language as tl

def pattern(conv2d_1, in_2, in_3, in_5, in_4, dtype):
    """Pattern: Conv2D (no bias) + BatchNorm + ReLU + Type conversion"""
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
    to = tmp_14.to(dtype)
    return to

def replacement_args(conv2d_1, in_2, in_3, in_5, in_4, dtype):
    return (conv2d_1, in_2, in_3, in_5, in_4, dtype)

@triton.jit
def fused_bn_relu_type_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch, channels, height, width,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process one element per thread
    channel = pid % channels
    spatial_idx = pid // channels
    
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Calculate linear indices
    input_idx = (spatial_idx * channels) + channel
    mean_idx = channel
    var_idx = channel
    weight_idx = channel
    bias_idx = channel
    output_idx = input_idx
    
    # Load values
    x = tl.load(input_ptr + input_idx)
    mean = tl.load(running_mean_ptr + mean_idx)
    var = tl.load(running_var_ptr + var_idx)
    weight = tl.load(weight_ptr + weight_idx)
    bias = tl.load(bias_ptr + bias_idx)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    var_plus_eps = var + eps
    sqrt_var_plus_eps = tl.sqrt(var_plus_eps)
    normalized = (x - mean) / sqrt_var_plus_eps
    bn_output = normalized * weight + bias
    
    # ReLU
    relu_output = tl.maximum(bn_output, 0.0)
    
    # Store result (for float16 conversion)
    tl.store(output_ptr + output_idx, relu_output)

@torch.fx.wrap
def fused_bn_relu_type(input, running_mean, running_var, weight, bias, eps=1e-05, dtype=torch.float16):
    batch, channels, height, width = input.shape
    output = torch.empty(input.shape, dtype=dtype, device=input.device)
    
    input_flat = input.reshape(-1)
    running_mean_flat = running_mean.reshape(-1)
    running_var_flat = running_var.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1)
    output_flat = output.reshape(-1)
    
    grid_size = batch * channels * height * width
    fused_bn_relu_type_kernel[(grid_size,)](
        input_flat, running_mean_flat, running_var_flat, weight_flat, bias_flat,
        output_flat, batch, channels, height, width, eps
    )
    
    return output

def replacement_func():
    return fused_bn_relu_type