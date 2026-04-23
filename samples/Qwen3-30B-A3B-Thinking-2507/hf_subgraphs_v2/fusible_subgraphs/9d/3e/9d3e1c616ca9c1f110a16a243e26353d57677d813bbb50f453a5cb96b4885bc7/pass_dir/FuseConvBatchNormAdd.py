import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def fused_conv_batchnorm_add(
    input,  # Input tensor: [B, C_in, H, W]
    weight,  # Conv weight: [C_out, C_in, 1, 1]
    mean,  # Mean: [C_out]
    var,  # Variance: [C_out]
    weight_bn,  # BatchNorm weight: [C_out]
    bias,  # BatchNorm bias: [C_out]
    add_input,  # Add input: [B, C_out, H, W]
    output,  # Output: [B, C_out, H, W]
    B, C_in, C_out, H, W,
    eps: tl.constexpr = 1e-05
):
    # Get the current index in the grid: this will iterate over B * H * W
    idx = tl.program_id(0)
    b = idx // (H * W)
    h = (idx % (H * W)) // W
    w = (idx % (H * W)) % W
    
    # Get the current channel (c) from the thread
    c = tl.thread_id(0)  
    
    # If channel index is beyond C_out, return (out of bounds)
    if c >= C_out:
        return
    
    # Compute the conv2d output for this (b, c, h, w)
    conv_val = 0.0
    for in_c in range(C_in):
        # Load input at (b, in_c, h, w)
        input_val = tl.load(input + b * C_in * H * W + in_c * H * W + h * W + w)
        # Load weight at (c, in_c, 0, 0)
        weight_val = tl.load(weight + c * C_in * 1 * 1 + in_c * 1 * 1)
        conv_val += input_val * weight_val
    
    # Apply BatchNorm
    mean_val = mean[c]
    var_val = var[c]
    weight_val_bn = weight_bn[c]
    bias_val = bias[c]
    
    # Compute denominator: sqrt(var_val + eps)
    denominator = tl.sqrt(var_val + eps)
    # Normalize and scale
    normalized = (conv_val - mean_val) / denominator
    scaled = normalized * weight_val_bn
    batch_normed = scaled + bias_val
    
    # Add the add_input
    add_val = tl.load(add_input + b * C_out * H * W + c * H * W + h * W + w)
    result = batch_normed + add_val
    
    # Store output
    tl.store(output + b * C_out * H * W + c * H * W + h * W + w, result)

@torch.fx.wrap
def fused_conv_batchnorm_add_wrapper(input, weight, mean, var, weight_bn, bias, add_input):
    B, C_in, H, W = input.shape
    _, C_out, _, _ = add_input.shape
    
    output = torch.empty_like(add_input)
    
    # Set block size to C_out
    block_size = C_out
    # Grid size is number of spatial positions * batches = B * H * W
    grid_size = B * H * W
    
    fused_conv_batchnorm_add[(grid_size, block_size)](
        input, weight, mean, var, weight_bn, bias, add_input, output,
        B, C_in, C_out, H, W,
        eps=1e-05
    )
    
    return output

def replacement_func():
    return fused_conv_batchnorm_add_wrapper