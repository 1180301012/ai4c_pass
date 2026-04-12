import torch
import triton
import triton.language as tl

# Pattern matching function for batch_norm + relu fusion
def pattern(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    # BatchNorm operation - matching exact argument order from model.py
    bn_output = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)
    # ReLU operation - matching exact argument order from model.py  
    relu_output = torch.nn.functional.relu(bn_output, inplace=False)
    return bn_output, relu_output

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)

# Triton kernel for fused BatchNorm + ReLU
@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr,
    channels, height, width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = channels * height * width
    mask = pid < total_elements
    
    # Load data and parameters
    input_data = tl.load(input_ptr + pid, mask=mask, other=0.0)
    running_mean_data = tl.load(running_mean_ptr + pid % channels, mask=(pid % channels) < channels, other=0.0)
    running_var_data = tl.load(running_var_ptr + pid % channels, mask=(pid % channels) < channels, other=1.0)
    weight_data = tl.load(weight_ptr + pid % channels, mask=(pid % channels) < channels, other=1.0)
    bias_data = tl.load(bias_ptr + pid % channels, mask=(pid % channels) < channels, other=0.0)
    
    # Apply batch norm formula
    centered = input_data - running_mean_data
    scaled = centered / tl.sqrt(running_var_data + eps)
    bn_result = scaled * weight_data + bias_data
    
    # Apply ReLU activation
    relu_result = tl.maximum(bn_result, 0.0)
    
    # Store result
    tl.store(out_ptr + pid, relu_result, mask=mask)

# Kernel wrapper for fused BatchNorm + ReLU
@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size, channels, height, width = input_tensor.shape
    total_elements = batch_size * channels * height * width
    
    # Create output tensor
    output = torch.empty((batch_size, channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_batchnorm_relu_kernel[grid_size](
        input_tensor, 
        running_mean, 
        running_var, 
        weight, 
        bias, 
        output,
        channels, height, width,
        eps, BLOCK_SIZE
    )
    
    return input_tensor, output

# Replacement function
def replacement_func():
    return fused_batchnorm_relu