import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for batch normalization"""
    # Program identifier for 1D grid
    pid = tl.program_id(0)
    element_start = pid * BLOCK_SIZE
    element_offsets = element_start + tl.arange(0, BLOCK_SIZE)
    element_mask = element_offsets < (n_rows * n_features)
    
    # Convert element offsets to row/feature indices
    feature_idx = element_offsets % n_features
    row_idx = element_offsets // n_features
    
    # Load input values for all elements in this block
    x_vals = tl.load(x_ptr + element_offsets, mask=element_mask, other=0.0)
    
    # Apply batch normalization formula - get parameters for each feature index
    mean_current = tl.load(mean_ptr + feature_idx, mask=element_mask, other=0.0)
    var_current = tl.load(var_ptr + feature_idx, mask=element_mask, other=1.0)
    weight_current = tl.load(weight_ptr + feature_idx, mask=element_mask, other=1.0)
    bias_current = tl.load(bias_ptr + feature_idx, mask=element_mask, other=0.0)
    
    # Convert to float32 for calculations
    x_float = x_vals.to(tl.float32)
    
    # Normalize element-wise
    denominator = tl.sqrt(var_current + eps)
    normalized = (x_float - mean_current) / denominator * weight_current + bias_current
    
    # Store result
    tl.store(out_ptr + element_offsets, normalized, mask=element_mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    """Wrapper function that launches the optimized batch normalization kernel"""
    
    # Get tensor dimensions
    n_rows, n_features = x.shape
    
    # Batch normalization kernel launch configuration
    BLOCK_SIZE = 64
    grid_size = ((n_rows * n_features) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Launch kernel - need to pass grid as tuple
    batch_norm_kernel[(grid_size,)](
        x, running_mean, running_var, weight, bias, out,
        n_rows, n_features, 1e-05, BLOCK_SIZE
    )
    
    return out

def pattern(in_7, in_0, in_1, in_3, in_2):
    """
    Pattern matching for batch normalization: torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    """
    return torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    """Extract arguments needed for the replacement"""
    return (in_7, in_0, in_1, in_3, in_2)

def replacement_func():
    """Return the replacement function"""
    return optimized_batch_norm