import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    """Match batch_norm followed by relu"""
    # Batch norm with exact arguments from the model
    bn_result = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, momentum, eps)
    # ReLU to match the model pattern
    relu_result = torch.nn.functional.relu(bn_result, inplace=False)
    return bn_result, relu_result

def replacement_args(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, momentum, eps)

@triton.jit
def fused_batch_norm_relu_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C, H, W,
    momentum, eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    """Fused BatchNorm + ReLU kernel"""
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Ensure bounds
    if pid_n >= N or pid_c >= C:
        return
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + pid_c)
    var = tl.load(running_var_ptr + pid_c)
    gamma = tl.load(weight_ptr + pid_c) if weight_ptr is not None else 1.0
    beta = tl.load(bias_ptr + pid_c) if bias_ptr is not None else 0.0
    
    # Compute batch norm normalization factor
    denom = tl.math.rsqrt(var + eps)
    
    # Process spatial dimensions for this batch and channel
    offset_base = pid_n * (C * H * W) + pid_c * (H * W)
    
    for i in range(H * W):
        offset = offset_base + i
        
        # Load input value
        x = tl.load(input_ptr + offset)
        
        # Batch norm: y = gamma * (x - mean) * sigma + beta
        # where sigma = 1 / sqrt(var + eps)
        bn_result = gamma * (x - mean) * denom + beta
        
        # ReLU activation
        relu_result = tl.maximum(0.0, bn_result)
        
        # Store result
        tl.store(output_ptr + offset, relu_result)

@torch.fx.wrap
def fused_batch_norm_relu(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
    """Optimized fused batch_norm + relu function"""
    N, C, H, W = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((N, C, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64  # Process multiple channels per program
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    fused_batch_norm_relu_kernel[(grid_n, grid_c)](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N, C, H, W,
        momentum,
        eps,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return fused_batch_norm_relu