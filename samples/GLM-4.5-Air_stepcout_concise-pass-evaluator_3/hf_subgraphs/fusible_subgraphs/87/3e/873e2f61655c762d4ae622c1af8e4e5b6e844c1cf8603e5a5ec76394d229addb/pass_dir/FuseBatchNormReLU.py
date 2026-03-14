import torch
import triton
import triton.language as tl

# Pattern for BatchNorm + ReLU fusion
def pattern(input_tensor, running_mean, running_var, weight, bias):
    normalized = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    output = torch.nn.functional.relu(normalized, inplace=True)
    return output

# Extract arguments for the fused kernel
def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

# Triton kernel for fused BatchNorm + ReLU
@triton.jit
def fused_batchnorm_relu_kernel(
    x_ptr,           # Input tensor (B, C, H, W)
    running_mean_ptr, # Running mean (C,)
    running_var_ptr,  # Running variance (C,)
    weight_ptr,       # BatchNorm weight (C,)
    bias_ptr,         # BatchNorm bias (C,)
    out_ptr,          # Output tensor (B, C, H, W)
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID - use 3D grid: batch, spatial, channels
    batch_id = tl.program_id(0)
    linear_spatial = tl.program_id(1)  # Combined H and W
    c_out = tl.program_id(2)
    
    # Decompose spatial coordinates
    h_out = linear_spatial // W
    w_out = linear_spatial % W
    
    # Create mask to check if coordinates are within bounds
    mask = (h_out < H) & (w_out < W) & (batch_id < B) & (c_out < C)
    
    # Calculate input offset
    input_offset = batch_id * C * H * W + c_out * H * W + h_out * W + w_out
    
    # Load input value
    x = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
    
    # Load BatchNorm parameters
    param_mask = (c_out < C)
    mean = tl.load(running_mean_ptr + c_out, mask=param_mask, other=0.0)
    var = tl.load(running_var_ptr + c_out, mask=param_mask, other=1.0)
    weight_val = tl.load(weight_ptr + c_out, mask=param_mask, other=1.0)
    bias_val = tl.load(bias_ptr + c_out, mask=param_mask, other=0.0)
    
    # BatchNorm computation: (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (x - mean) / tl.sqrt(var + eps) * weight_val + bias_val
    
    # ReLU activation: max(0, normalized)
    output = tl.maximum(normalized, 0.0)
    
    # Store result
    out_offset = batch_id * C * H * W + c_out * H * W + h_out * W + w_out
    tl.store(out_ptr + out_offset, output, mask=mask)

@torch.fx.wrap
def fused_batchnorm_relu(input_tensor, running_mean, running_var, weight, bias):
    B, C, H, W = input_tensor.shape
    
    out = torch.empty((B, C, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Configure block size for GPU occupancy
    BLOCK_SIZE = 128
    
    # Calculate number of spatial programs (H * W)
    spatial_size = H * W
    
    # Launch kernel with 3D grid: batch, spatial, channels
    fused_batchnorm_relu_kernel[(B, spatial_size, C,)](
        input_tensor, running_mean, running_var, weight, bias, out,
        B, C, H, W,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_batchnorm_relu