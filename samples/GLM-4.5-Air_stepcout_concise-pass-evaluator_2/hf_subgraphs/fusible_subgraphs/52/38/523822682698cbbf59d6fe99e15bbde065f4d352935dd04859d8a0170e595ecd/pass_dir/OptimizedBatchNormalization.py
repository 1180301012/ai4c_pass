import torch
import triton
import triton.language as tl

# Pattern matching function for batch normalization
def pattern(x, running_mean, running_var, weight, bias):
    """Match batch normalization operation"""
    result = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return result

# Argument extraction function
def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

# Triton kernel for batch normalization - simpler and more efficient
@triton.jit
def batch_norm_kernel_simple(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    n_features,
    eps: float,
):
    # Each program processes one element: x[batch_idx, feature_idx]
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Check bounds
    if batch_idx >= batch_size or feature_idx >= n_features:
        return
    
    # Calculate memory index
    input_index = batch_idx * n_features + feature_idx
    
    # Load normalization parameters for this feature
    running_mean = tl.load(running_mean_ptr + feature_idx)
    running_var = tl.load(running_var_ptr + feature_idx)
    weight_val = tl.load(weight_ptr + feature_idx)
    bias_val = tl.load(bias_ptr + feature_idx)
    
    # Load input
    x = tl.load(x_ptr + input_index)
    
    # Apply batch normalization
    std = tl.sqrt(running_var + eps)
    inv_std = 1.0 / std
    x_norm = (x - running_mean) * inv_std
    x_scaled = x_norm * weight_val
    
    # Add bias and store
    out = x_scaled + bias_val
    tl.store(out_ptr + input_index, out)

# Kernel wrapper
@torch.fx.wrap
def batch_norm_fused(x, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization using Triton
    Simple and efficient element-wise processing for better performance
    """
    batch_size = x.shape[0]
    n_features = x.shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Flatten input tensor for kernel (row-major order)
    x_flat = x.view(-1)  # [batch_size * n_features]
    
    # Launch kernel with one program per output element
    batch_norm_kernel_simple[(batch_size, n_features)](
        x_ptr=x_flat,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        n_features=n_features,
        eps=1e-05,
    )
    
    return out

# Replacement function
def replacement_func():
    return batch_norm_fused