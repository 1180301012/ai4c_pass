import torch
import triton
import triton.language as tl

# Pattern matching function for batch norm operation
def pattern(in_7, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7

# Argument extraction function
def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

# Optimized batch norm kernel using Triton
@triton.jit
def batch_norm_kernel(
    x_ptr,        # input [batch_size, features]
    mean_ptr,     # running mean [features]
    var_ptr,      # running var [features] 
    weight_ptr,   # weight/scale [features]
    bias_ptr,     # bias/shift [features]
    out_ptr,      # output [batch_size, features]
    batch_size,
    features,
):
    # Program identifier
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # features dimension
    
    # Check ranges
    if pid_m >= batch_size or pid_n >= features:
        return
    
    # Compute pointer addresses
    x_base = x_ptr + pid_m * features + pid_n
    mean_base = mean_ptr + pid_n
    var_base = var_ptr + pid_n
    weight_base = weight_ptr + pid_n
    bias_base = bias_ptr + pid_n
    out_base = out_ptr + pid_m * features + pid_n
    
    # Load parameters
    x_val = tl.load(x_base)
    mean_val = tl.load(mean_base)
    var_val = tl.load(var_base)
    weight_val = tl.load(weight_base)
    bias_val = tl.load(bias_base)
    
    # Apply batch normalization: y = (x - mean) * weight / sqrt(var + epsilon) + bias
    epsilon = 1e-05
    normalized = (x_val - mean_val) / tl.sqrt(var_val + epsilon)
    result = normalized * weight_val + bias_val
    
    # Store result
    tl.store(out_base, result)

# Kernel wrapper
@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    batch_size, features = x.shape
    
    # Calculate grid dimensions - launch one program per output element
    grid_m = batch_size
    grid_n = features
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty((batch_size, features), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    batch_norm_kernel[grid](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        features=features,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_batch_norm