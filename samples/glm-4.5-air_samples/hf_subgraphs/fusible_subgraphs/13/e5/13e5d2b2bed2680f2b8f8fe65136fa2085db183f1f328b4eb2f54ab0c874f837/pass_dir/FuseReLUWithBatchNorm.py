import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, running_mean, running_var, weight, bias):
    # Simple batch norm pattern (without ReLU for now)
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

# Argument extraction function  
def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

# Optimized kernel with simple 1D tiling for better performance
@triton.jit
def batch_norm_kernel_simple(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for 1D tiling
    pid = tl.program_id(0)
    
    # Use tl.arange with compile-time constant size
    offsets_in_block = tl.arange(0, BLOCK_SIZE)
    
    # Compute start offset for this program
    offset = pid * BLOCK_SIZE
    
    # Create mask for valid elements
    mask = offset + offsets_in_block < n_elements
    
    # Calculate actual offsets
    actual_offsets = offset + offsets_in_block
    
    # Load input data
    x = tl.load(x_ptr + actual_offsets, mask=mask, other=0.0)
    
    # Convert offsets to indices for parameter loading
    flat_idx = actual_offsets
    feature_idx = flat_idx % n_features
    
    # Load batch norm parameters using feature index
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=feature_idx < n_features, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=feature_idx < n_features, other=0.0)
    weight = tl.load(weight_ptr + feature_idx, mask=feature_idx < n_features, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, mask=feature_idx < n_features, other=0.0)
    
    # Apply batch normalization formula: (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    variance = running_var + eps
    std = tl.sqrt(variance)
    norm_x = (x - running_mean) / std
    output = norm_x * weight + bias
    
    # Apply ReLU: max(0, output)
    fused_output = tl.maximum(output, 0.0)
    
    # Store result with masking
    tl.store(out_ptr + actual_offsets, fused_output, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(x, running_mean, running_var, weight, bias):
    # Get input tensor shape
    n_elements = x.numel()
    n_features = x.shape[-1]  # Last dimension is features (128)
    n_batch = n_elements // n_features
    
    # Use optimal block size for this problem size (1000 elements total)
    BLOCK_SIZE = 128  # Optimal block size for this workload
    
    # Calculate grid dimensions (1D grid)
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    batch_norm_kernel_simple[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        n_elements,
        n_features,
        BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_relu_batch_norm