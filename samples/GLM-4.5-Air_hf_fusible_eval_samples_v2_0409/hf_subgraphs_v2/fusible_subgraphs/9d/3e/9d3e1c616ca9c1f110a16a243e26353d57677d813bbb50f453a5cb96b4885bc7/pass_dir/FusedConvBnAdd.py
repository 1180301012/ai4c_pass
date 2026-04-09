import torch
import triton
import triton.language as tl

# Pattern 1: Conv2D(in_6, in_4) + BatchNorm(in_0, in_1, in_3, in_2) + Add(in_5)
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Extract the tensors in the order they'll be used in the fused kernel
    # input_tensor for conv = in_6, conv_weight = in_4
    # batch_norm: running_mean = in_0, running_var = in_1, weight = in_3, bias = in_2
    # add_tensor = in_5
    return (in_6, in_4, None, in_0, in_1, in_3, in_2, 0.1, 1e-05, False, in_5)

# Triton kernel for fused Conv2D + BatchNorm + Add
@triton.jit
def fused_conv_bn_add_kernel(
    input_ptr, conv_weight_ptr, bn_weight_ptr, bn_bias_ptr, running_mean_ptr, running_var_ptr,
    add_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location across all batches and output channels
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate spatial coordinates
    spatial_offset = pid % (height * width)
    h = spatial_offset // width
    w = spatial_offset % width
    
    # Calculate batch and output channel indices
    batch = (pid // (height * width)) // out_channels
    oc = (pid // (height * width)) % out_channels
    
    # Check bounds
    if batch >= batch_size or oc >= out_channels or h >= height or w >= width:
        return
    
    # Load parameters for this output channel
    weight_val = tl.load(conv_weight_ptr + oc * in_channels + 0)  # 1x1 conv, so in_channels=0
    weight_val = weight_val.to(tl.float32)
    
    bn_w = tl.load(bn_weight_ptr + oc)
    bn_b = tl.load(bn_bias_ptr + oc)
    mean = tl.load(running_mean_ptr + oc)
    var = tl.load(running_var_ptr + oc)
    
    # Add numerical stability
    var = tl.maximum(var, 1e-5)
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Apply fused batch norm: (x - mean) * (weight * rstd) + bias
    gamma = bn_w * rstd
    beta = bn_b - mean * gamma
    
    # Load input data (1x1 convolution means we spatially multiply)
    for c in range(in_channels):
        input_idx = batch * in_channels * height * width + c * height * width + h * width + w
        input_val = tl.load(input_ptr + input_idx)
        input_val = input_val.to(tl.float32)
        
        # Convolution (1x1, simple channel multiplication)
        conv_val = input_val * weight_val
        
        # Batch normalization and addition
        output_val = conv_val * gamma + beta
        
        # Load add tensor
        add_idx = batch * out_channels * height * width + oc * height * width + h * width + w
        add_val = tl.load(add_ptr + add_idx)
        add_val = add_val.to(tl.float32)
        
        # Final addition
        result = output_val + add_val
        
        # Store result
        out_idx = batch * out_channels * height * width + oc * height * width + h * width + w
        tl.store(out_ptr + out_idx, result.to(input_val.dtype))

@torch.fx.wrap
def fused_conv_bn_add(input_tensor, conv_weight, conv_bias, running_mean, running_var, bn_weight, bn_bias, momentum, eps, training, add_tensor):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, _, _ = conv_weight.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate total number of programs needed
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 128  # Can be tuned
    
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_conv_bn_add_kernel[grid](
        input_tensor,
        conv_weight,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        add_tensor,
        output,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_bn_add