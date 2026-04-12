import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must match batch_norm + silu sequence exactly
def pattern(input_tensor, running_mean, running_var, bias, weight):
    batch_norm_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_out = torch.nn.functional.silu(batch_norm_out, inplace=True)
    return silu_out  # Return the silu output

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, bias, weight):
    return (input_tensor, running_mean, running_var, bias, weight)

# Triton kernel for fused batch_norm + silu operation
@triton.jit
def fused_batch_norm_silu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    spatial_size: tl.constexpr,
    num_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    # Load feature indices for this block
    feature_idx = (offsets // spatial_size) % num_features
    spatial_idx = offsets % spatial_size
    
    # Load batch norm parameters for current features
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=feature_idx < num_features, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=feature_idx < num_features, other=0.0)
    weight = tl.load(weight_ptr + feature_idx, mask=feature_idx < num_features, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, mask=feature_idx < num_features, other=0.0)
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Batch norm computation: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    inv_std = tl.math.rsqrt(running_var + eps)
    normalized = (input_vals - running_mean) * inv_std
    batch_norm_out = normalized * weight + bias
    
    # SiLU activation: y = x * sigmoid(x)
    silu_out = batch_norm_out * (1.0 / (1.0 + tl.math.exp(-batch_norm_out)))
    
    # Store result
    tl.store(output_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batch_norm_silu(input_tensor, running_mean, running_var, bias, weight):
    # Get tensor dimensions
    spatial_size = input_tensor.numel() // input_tensor.size(1)  # Total spatial elements
    num_features = input_tensor.size(1)  # Number of channels (512)
    
    # Set block size and compute grid size
    BLOCK_SIZE = 1024
    num_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    fused_batch_norm_silu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        spatial_size=spatial_size,
        num_features=num_features,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_batch_norm_silu