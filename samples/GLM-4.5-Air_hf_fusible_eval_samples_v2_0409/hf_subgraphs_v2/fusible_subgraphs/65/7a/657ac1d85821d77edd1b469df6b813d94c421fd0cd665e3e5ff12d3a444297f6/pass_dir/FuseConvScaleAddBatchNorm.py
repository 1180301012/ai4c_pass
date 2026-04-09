import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scale_factor, running_mean, running_var, weight_bn, bias_bn, shortcut_input):
    """
    Pattern matching: Conv2D → Element-wise Multiply → Add → BatchNorm
    Optimizes the main computational path by fusing operations
    """
    # Conv2D operation 
    conv2d_result = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Element-wise multiplication with layer scale
    scale_result = conv2d_result * scale_factor
    
    # Shortcut addition
    add_result = shortcut_input + scale_result
    
    # Batch normalization
    bn_result = torch.nn.functional.batch_norm(add_result, running_mean, running_var, weight_bn, bias_bn, False, 0.1, 1e-05)
    
    return (bn_result, add_result)

def replacement_args(conv_input, conv_weight, conv_bias, scale_factor, running_mean, running_var, weight_bn, bias_bn, shortcut_input):
    return (conv_input, conv_weight, conv_bias, scale_factor, running_mean, running_var, weight_bn, bias_bn, shortcut_input)

@triton.jit
def fused_conv_scale_add_bnorm_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr,
    running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    shortcut_ptr, out_ptr, shortcut_out_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w, groups, eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Fused kernel: Conv2D → Scale → Add → BatchNorm"""
    
    # Grid setup  
    pid_m = tl.program_id(0)  # batch * height * width
    pid_n = tl.program_id(1)  # out_channels
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Get coordinates
    linear_idx = pid_m
    batch_id = linear_idx // (out_height * out_width)
    spatial_idx = linear_idx % (out_height * out_width)
    h_id = spatial_idx // out_width
    w_id = spatial_idx % out_width
    
    # Initialize accumulator
    acc = 0.0
    
    # Load bias and scale
    bias_val = tl.load(bias_ptr + pid_n, mask=(pid_n < out_channels), other=0.0)
    scale_val = tl.load(scale_ptr + pid_n, mask=(pid_n < out_channels), other=1.0)
    
    # Convolution computation
    for kc in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input coordinates
                ih_in = h_id * stride_h - pad_h + kh  
                iw_in = w_id * stride_w - pad_w + kw
                
                # Check bounds
                if ih_in >= 0 and ih_in < in_height and iw_in >= 0 and iw_in < in_width:
                    # Compute indices
                    input_idx = batch_id * in_channels * in_height * in_width + \
                               kc * in_height * in_width + \
                               ih_in * in_width + iw_in
                    
                    weight_idx = pid_n * in_channels * kernel_h * kernel_w + \
                                kc * kernel_h * kernel_w + \
                                kh * kernel_w + kw
                    
                    # Load values
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Apply bias and scaling
    acc = bias_val + acc * scale_val
    
    # Batch normalization
    running_mean_val = tl.load(running_mean_ptr + pid_n, mask=(pid_n < out_channels), other=0.0) 
    running_var_val = tl.load(running_var_ptr + pid_n, mask=(pid_n < out_channels), other=1.0)
    weight_bn_val = tl.load(weight_bn_ptr + pid_n, mask=(pid_n < out_channels), other=1.0)
    bias_bn_val = tl.load(bias_bn_ptr + pid_n, mask=(pid_n < out_channels), other=0.0)
    
    inv_std = 1.0 / tl.sqrt(running_var_val + eps)
    acc_normalized = bias_bn_val + weight_bn_val * (acc - running_mean_val) * inv_std
    
    # Get shortcut input value
    shortcut_idx = batch_id * out_channels * out_height * out_width + \
                   pid_n * out_height * out_width + \
                   h_id * out_width + w_id
    
    shortcut_val = tl.load(shortcut_ptr + shortcut_idx)
    
    # Add shortcut
    final_result = shortcut_val + acc_normalized
    
    # Store results
    output_idx = batch_id * out_channels * out_height * out_width + \
                 pid_n * out_height * out_width + \
                 h_id * out_width + w_id
    
    tl.store(out_ptr + output_idx, final_result)
    tl.store(shortcut_out_ptr + output_idx, acc_normalized)

@torch.fx.wrap
def fused_conv_scale_add_bnorm_optimized(conv_input, conv_weight, conv_bias, scale_factor, 
                                        running_mean, running_var, weight_bn, bias_bn, shortcut_input):
    """Optimized fused operation"""
    
    # Get shapes
    batch_size, in_channels, in_height, in_width = conv_input.shape
    out_channels, _, kernel_h, kernel_w = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2*0 - kernel_h) // 1 + 1
    out_width = (in_width + 2*0 - kernel_w) // 1 + 1
    
    # Create output tensors
    output_shape = (batch_size, out_channels, out_height, out_width)
    result = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    shortcut_result = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Grid configuration
    total_spatial = batch_size * out_height * out_width
    grid_m = (total_spatial + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv_scale_add_bnorm_kernel[(
        grid_m, grid_n
    )](
        conv_input, conv_weight, conv_bias, scale_factor,
        running_mean, running_var, weight_bn, bias_bn,
        shortcut_input, result, shortcut_result,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_h, kernel_w,
        1, 1, 0, 0, 1, 1e-05,
        128, 64  # BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return result, shortcut_result

def replacement_func():
    return fused_conv_scale_add_bnorm_optimized