import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(a, b, c):
    # Simplified pattern: conv + add
    tmp_1 = torch.conv2d(a, b, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = tmp_1 + c
    return tmp_2

# Argument extraction function  
def replacement_args(a, b, c):
    return (a, b, c)

# Optimized Triton kernel
@triton.jit
def fused_conv_bn_add_kernel(
    input_ptr, 
    weight_ptr, 
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    add_input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program id for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Pre-compute scale and bias parameters
    channel_idx = pid_n
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    bn_weight_val = tl.load(bn_weight_ptr + channel_idx)
    bn_bias_val = tl.load(bn_bias_ptr + channel_idx)
    
    # Compute fused parameters: scale = weight / sqrt(var + eps), bias = bias - mean * scale
    inv_std = tl.math.rsqrt(running_var + eps)
    fused_scale = bn_weight_val * inv_std
    fused_bias = bn_bias_val - running_mean * fused_scale
    
    # Process output spatial tiles
    for h_offset in range(0, height, 1):
        for w_offset in range(0, width, 1):
            # Compute output position
            out_h = h_offset
            out_w = w_offset
            
            # Only process if within bounds
            if out_h < height and out_w < width:
                # Initialize accumulator for this output pixel
                result = 0.0
                
                # Convolution computation (1x1 kernel, so no spatial loop needed)
                for c_in in range(in_channels):
                    # Load input value
                    in_offset = (pid_m * in_channels + c_in) * height * width + out_h * width + out_w
                    input_val = tl.load(input_ptr + in_offset)
                    
                    # Load weight value  
                    weight_offset = channel_idx * in_channels + c_in
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    # Apply fused weight and accumulate
                    result += input_val * weight_val
                
                # Apply fused batch norm and add
                result = result * fused_scale + fused_bias
                
                # Load and add the input to the batch
                add_offset = (pid_m * out_channels + channel_idx) * height * width + out_h * width + out_w
                add_val = tl.load(add_input_ptr + add_offset)
                
                result = result + add_val
                
                # Store result
                out_offset = (pid_m * out_channels + channel_idx) * height * width + out_h * width + out_w
                tl.store(output_ptr + out_offset, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_bn_add(input, weight, running_mean, running_var, bn_weight, bn_bias, add_input):
    b, in_c, h, w = input.shape
    out_c = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((b, out_c, h, w), dtype=input.dtype, device=input.device)
    
    # Set up launch configuration
    grid_m = b
    grid_n = out_c
    grid = (grid_m, grid_n)
    
    # Launch Triton kernel
    fused_conv_bn_add_kernel[grid](
        input, weight, running_mean, running_var, bn_weight, bn_bias,
        add_input, output, b, in_c, out_c, h, w,
        eps=1e-05
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_bn_add