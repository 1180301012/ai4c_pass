import torch
import triton
import triton.language as tl

def pattern(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    """Conv2D + BatchNorm + LeakyReLU pattern - exact match to model"""
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return tmp_8

def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)



@torch.fx.wrap
def fused_conv2d_bn_relu_add(input_tensor, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, skip_connection):
    """Fused Conv2D + BatchNorm + LeakyReLU + Addition using Triton"""
    # Get tensor dimensions
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, KH, KW = conv_weight.shape
    H_out, W_out = H_in, W_in  # stride=1, padding=1
    
    # Allocate output tensor - this is an allowed operation
    output = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton fused kernel implementation for all supported dtypes
    # Auto-tune kernel launch parameters based on input dimensions
    block_size_configs = [
        (8, 4, 32),    # Small config
        (16, 8, 64),   # Medium config  
        (32, 16, 128), # Large config
    ]
    
    # Choose configuration based on output tensor size
    total_output_elements = N * C_out * H_out * W_out
    if total_output_elements < 1024 * 1024:  # Small tensor
        config_idx = 0
    elif total_output_elements < 1024 * 1024 * 8:  # Medium tensor
        config_idx = 1  
    else:  # Large tensor
        config_idx = 2
    
    block_m, block_n, block_k = block_size_configs[config_idx]
    
    # Optimized kernel launch
    grid_size = (total_output_elements + block_m * block_n - 1) // (block_m * block_n)
    fused_conv_bn_relu_kernel[(
        grid_size,
    )](
        input_tensor, conv_weight, bn_running_mean, bn_running_var,
        bn_weight, bn_bias, skip_connection, output,
        N, C_out, C_in, H_in, W_in, H_out, W_out, KH, KW, 
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k
    )
    
    return output

@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
    weight_bn_ptr, bias_bn_ptr, skip_connection_ptr,
    output_ptr,
    N, C_out, C_in, H_in, W_in, H_out, W_out, KH, KW,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Fused Conv2D + BatchNorm + LeakyReLU kernel"""
    pid = tl.program_id(0)
    total_elements = N * C_out * H_out * W_out
    if pid >= total_elements:
        return
    
    # Calculate indices
    batch = pid // (C_out * H_out * W_out)
    remaining = pid % (C_out * H_out * W_out)
    out_channel = remaining // (H_out * W_out)
    remaining = remaining % (H_out * W_out)
    h_out = remaining // W_out
    w_out = remaining % W_out
    
    # Initialize accumulator for convolution
    accumulator = 0.0
    
    # 3x3 convolution with stride=1, padding=1
    for kh in range(KH):
        for kw in range(KW):
            h_in = h_out + kh - 1  # padding=1
            w_in = w_out + kw - 1  # padding=1
            if (h_in >= 0 and h_in < H_in) and (w_in >= 0 and w_in < W_in):
                input_val = tl.load(input_ptr + batch * C_in * H_in * W_in + 
                                   h_in * W_in + w_in)
                weight_val = tl.load(weight_ptr + 
                                    out_channel * C_in * KH * KW + 
                                    kh * KW * C_in + kw * C_in)
                accumulator += input_val * weight_val
    
    # Load BatchNorm parameters
    bn_weight = tl.load(weight_bn_ptr + out_channel)
    bn_bias = tl.load(bias_bn_ptr + out_channel)
    running_mean = tl.load(running_mean_ptr + out_channel)
    running_var = tl.load(running_var_ptr + out_channel)
    
    # Apply BatchNorm (in float32 for precision)
    scale = bn_weight * (running_var + 1e-05).rsqrt()
    fused_output = (accumulator - running_mean) * scale + bn_bias
    
    # Apply LeakyReLU (negative slope = 0.01)
    fused_output = tl.where(fused_output >= 0, fused_output, fused_output * 0.01)
    
    # Add skip connection
    skip_val = tl.load(skip_connection_ptr + pid)
    final_output = fused_output + skip_val
    
    # Store result in original precision
    tl.store(output_ptr + pid, final_output)

def replacement_func():
    return fused_conv2d_bn_relu_add