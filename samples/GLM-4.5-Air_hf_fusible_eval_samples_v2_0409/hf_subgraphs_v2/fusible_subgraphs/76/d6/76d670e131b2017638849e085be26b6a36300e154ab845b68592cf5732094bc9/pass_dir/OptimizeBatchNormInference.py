import torch
import triton
import triton.language as tl

# Pattern matching for batch normalization
def pattern(in_7, in_0, in_1, in_3, in_2):
    """
    Match the batch normalization pattern: 
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    """
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7

# Argument extraction function
def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

# Triton kernel for optimized batch normalization (inference only)
@triton.jit
def batch_norm_kernel(
    x_ptr, 
    mean_ptr, 
    var_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr, 
    batch_size, 
    num_features,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized batch normalization kernel for inference
    
    Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    Where eps = 1e-05
    """
    # Compute program id
    pid = tl.program_id(0)
    
    # Check if this program should be active (simplified bounds check)
    if pid >= batch_size * num_features:
        return
    
    # Compute offset for this program
    offset = pid
    
    # Load input data
    x = tl.load(x_ptr + offset).to(tl.float32)
    
    # Compute feature offset (each program handles one element)
    feature_offset = pid % num_features
    
    # Load normalization parameters
    mean = tl.load(mean_ptr + feature_offset).to(tl.float32)
    var = tl.load(var_ptr + feature_offset).to(tl.float32)
    weight = tl.load(weight_ptr + feature_offset).to(tl.float32)
    bias = tl.load(bias_ptr + feature_offset).to(tl.float32)
    
    # Apply batch normalization formula
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    var_with_eps = var + 1e-05
    sqrt_var = tl.sqrt(var_with_eps)
    normalized = (x - mean) / sqrt_var
    result = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offset, result.to(tl.float32))

# Kernel wrapper
@torch.fx.wrap
def optimized_batch_norm_inference(x, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization for inference (training=False)
    """
    batch_size, num_features = x.shape
    
    # Optimal block size for pointwise operations
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = ((batch_size * num_features + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor with same dtype as input
    out = torch.empty_like(x)
    
    # Launch kernel
    batch_norm_kernel[grid_size](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        num_features=num_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_batch_norm_inference