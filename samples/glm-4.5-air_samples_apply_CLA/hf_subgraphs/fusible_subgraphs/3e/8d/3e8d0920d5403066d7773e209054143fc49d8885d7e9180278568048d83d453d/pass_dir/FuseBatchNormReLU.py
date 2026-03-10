import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    # Match: batch_norm + relu pattern
    tmp_7 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8

def replacement_args(x, running_mean, running_var, weight, bias):
    return x, running_mean, running_var, weight, bias

@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_features,
    batch_size,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate batch and channel offsets
    batch_offset = pid_m * BLOCK_SIZE_M
    channel_offset = pid_n * BLOCK_SIZE_N
    
    # Ensure we don't go out of bounds
    if batch_offset >= batch_size:
        return
    
    # Process each channel in this thread
    for c in range(channel_offset, min(channel_offset + BLOCK_SIZE_N, num_features)):
        # Load normalization parameters
        mean = tl.load(running_mean_ptr + c)
        var = tl.load(running_var_ptr + c)
        gamma = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)
        
        # Compute epsilon for numerical stability
        epsilon = 0.001
        inv_std = 1.0 / tl.sqrt(var + epsilon)
        
        # Process each spatial position
        for h in range(0, height):
            for w in range(0, width):
                # Load input
                x_val = tl.load(x_ptr + batch_offset * num_features * height * width + 
                              c * height * width + h * width + w)
                
                # Batch normalization formula: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
                bn_val = gamma * (x_val - mean) * inv_std + beta
                
                # ReLU activation
                relu_val = tl.maximum(bn_val, 0.0)
                
                # Store result
                tl.store(out_ptr + batch_offset * num_features * height * width + 
                        c * height * width + h * width + w, relu_val)

@torch.fx.wrap
def fused_batch_norm_relu(x, running_mean, running_var, weight, bias):
    num_features = running_mean.shape[0]
    batch_size = x.shape[0]
    height = x.shape[2]
    width = x.shape[3]
    
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    
    # Use optimal block sizes for better GPU occupancy
    BLOCK_SIZE_M = 32  # Batch dimension
    BLOCK_SIZE_N = 64  # Channel dimension
    
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_M), 
            triton.cdiv(num_features, BLOCK_SIZE_N))
    
    fused_bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        num_features, batch_size, height, width,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return out

def replacement_func():
    return fused_batch_norm_relu