import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr, 
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    num_features,
    eps,
):
    """Simple batch norm kernel - one feature per program"""
    pid = tl.program_id(0)
    feature_idx = pid
    
    # Early return if out of bounds
    if feature_idx >= num_features:
        return
    
    # Load parameters for this feature (safe due to bounds check)
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=None)
    running_var = tl.load(running_var_ptr + feature_idx, mask=None)
    weight = tl.load(weight_ptr + feature_idx, mask=None)
    bias = tl.load(bias_ptr + feature_idx, mask=None)
    
    # Compute normalization constants
    inv_std = tl.rsqrt(running_var + eps)
    
    # Process all batch elements for this feature
    for i in range(batch_size):
        batch_offset = i * num_features
        
        # Load input feature for this batch element
        x_val = tl.load(x_ptr + batch_offset + feature_idx, mask=None)
        
        # Apply batch normalization formula
        # y = (x - running_mean) * inv_std * weight + bias
        normalized = (x_val - running_mean) * inv_std
        normalized = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + batch_offset + feature_idx, normalized, mask=None)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias, eps=1e-5):
    """Optimized batch normalization using Triton kernels"""
    batch_size, num_features = x.shape
    
    # Create output tensor as float32 for Triton computation, then convert back
    out = torch.empty((batch_size, num_features), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one program per feature
    grid_size = num_features
    
    # Launch kernel - simple 1D grid
    batch_norm_kernel[grid_size](
        x, running_mean, running_var, weight, bias, out,
        batch_size, num_features, eps
    )
    
    # Convert back to original dtype
    return out.to(x.dtype)

def pattern(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5, cudnn_enabled=True):
    """Match the batch norm operation pattern"""
    # The original call has the signature:
    # torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, 
    #                               training momentum, eps, cudnn_enabled)
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, 
                                        training, momentum, eps, cudnn_enabled)

def replacement_args(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5, cudnn_enabled=True):
    """Extract arguments for the replacement"""
    return (x, running_mean, running_var, weight, bias, eps)

def replacement_func():
    """Return the optimized batch norm function"""
    return optimized_batch_norm