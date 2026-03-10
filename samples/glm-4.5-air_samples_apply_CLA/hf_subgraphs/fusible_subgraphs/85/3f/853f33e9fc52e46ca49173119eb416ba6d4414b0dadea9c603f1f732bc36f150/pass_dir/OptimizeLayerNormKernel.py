import torch
import triton
import triton.language as tl

# Pattern matching function for layer normalization
def pattern(in_2, norm_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(in_2, norm_shape, weight, bias, eps)

# Argument extraction function
def replacement_args(in_2, norm_shape, weight, bias, eps):
    # Extract the actual normalized dimension size from the tuple
    normalized_dim = norm_shape[0]
    return (in_2, normalized_dim, weight, bias, eps)

# Optimized Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    n_elements,
    normalized_dim,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_start = row_idx * normalized_dim
    offsets = row_start + tl.arange(0, normalized_dim)
    mask = offsets < n_elements
    
    # Load the row of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, mask=mask) / normalized_dim
    
    # Compute variance
    x_centered = x - mean
    x_centered_sq = x_centered * x_centered
    var = tl.sum(x_centered_sq, mask=mask) / normalized_dim
    
    # Compute standard deviation (numerically stable)
    std = tl.sqrt(var + EPS)
    
    # Normalize
    x_normalized = x_centered / std
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Apply weight and bias
    out = x_normalized * weight + bias
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(mean_ptr + row_idx, mean)
    tl.store(var_ptr + row_idx, var)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    normalized_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_start = row_idx * normalized_dim
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < n_elements
    
    # Handle case where BLOCK_SIZE might be larger than normalized_dim
    if BLOCK_SIZE > normalized_dim:
        offsets = offsets % normalized_dim
        # Update mask for case where we wrapped around
        mask = (offsets < n_elements) & (offsets < normalized_dim)
    
    # Load the row of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean - first count valid elements and sum
    valid_mask = mask & (offsets < normalized_dim)
    valid_count = tl.where(valid_mask, 1, 0).to(tl.int32)
    n_valid = tl.sum(valid_count)
    
    x_sum = tl.sum(x)
    mean = x_sum / n_valid if n_valid > 0 else 0.0
    
    # Compute variance - handle division by zero
    if n_valid > 0:
        x_centered = x - mean
        x_centered_sq = x_centered * x_centered
        sum_sq = tl.sum(x_centered_sq)
        var = sum_sq / n_valid
    else:
        x_centered = x
        x_centered_sq = x * x
        var = tl.sum(x_centered_sq) / normalized_dim
    
    # Compute standard deviation
    std = tl.sqrt(var + EPS)
    
    # Normalize
    x_normalized = x_centered / std
    
    # Load weight and bias (ensure proper broadcasting)
    weight = tl.load(weight_ptr + offsets, mask=valid_mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=valid_mask, other=0.0)
    
    # Apply affine transformation
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=valid_mask)

@torch.fx.wrap
def optimized_layer_norm(x, normalized_dim, weight, bias, eps=1e-12):
    """
    Optimized layer normalization implementation using Triton
    """
    # Get input tensor shape
    input_shape = x.shape
    batch_size = input_shape[0]  # Should be 1 based on weight_meta
    seq_len = input_shape[1]     # Should be 197 based on weight_meta
    hidden_size = normalized_dim  # Should be 768 or 1024 based on weight_meta
    
    # Total number of elements
    n_elements = batch_size * seq_len * hidden_size
    
    # Number of rows (batch_size * seq_len)
    n_rows = batch_size * seq_len
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Set up grid for kernel launch - use power of 2 for BLOCK_SIZE
    BLOCK_SIZE = min(1024, normalized_dim)  # BLOCK_SIZE cannot exceed normalized_dim
    # Make BLOCK_SIZE a power of 2
    BLOCK_SIZE = 2 ** (BLOCK_SIZE.bit_length() - 1) if BLOCK_SIZE > 0 else 1
    grid = (n_rows,)
    
    # For better performance, we can optimize for common hidden sizes
    if hidden_size in [768, 1024]:
        # Use optimized block size for common transformer models
        if hidden_size == 768:
            BLOCK_SIZE = 512  # Power of 2 close to 768
        elif hidden_size == 1024:
            BLOCK_SIZE = 1024  # Power of 2, full warp utilization
        
        # Ensure BLOCK_SIZE doesn't exceed normalized_dim and is power of 2
        BLOCK_SIZE = min(BLOCK_SIZE, normalized_dim)
        BLOCK_SIZE = 2 ** (BLOCK_SIZE.bit_length() - 1) if BLOCK_SIZE > 0 else 1
    
    # Launch the optimized kernel
    optimized_layer_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        n_elements,
        normalized_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=eps,
    )
    
    return out

# Replacement function
def replacement_func():
    # Return a function that accepts all arguments
    def optimized_layer_norm_wrapper(x, normalized_dim, weight, bias, eps=1e-12):
        return optimized_layer_norm(x, normalized_dim, weight, bias, eps)
    return optimized_layer_norm_wrapper