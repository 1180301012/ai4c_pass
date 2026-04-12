import torch
import triton
import triton.language as tl
import math

@triton.jit
def layer_norm_kernel(
    x_ptr,                    # Input tensor pointer
    gamma_ptr,                # Weight pointer  
    beta_ptr,                 # Bias pointer
    out_ptr,                  # Output tensor pointer
    n_elements,               # Total number of elements
    feature_dim,              # Feature dimension size
    eps: tl.constexpr,        # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr, # Block size for computation
    REDUCE_BLOCK_SIZE: tl.constexpr, # Block size for reduction
):
    """High-performance layer normalization kernel using Triton"""
    # Each program handles one row (sequence position)
    pid = tl.program_id(0)
    stride = tl.program_id(1)
    
    # Calculate offset for this row
    row_offset = pid * feature_dim
    
    # Load current row data  
    row_start = x_ptr + row_offset
    mask = tl.arange(0, feature_dim) < feature_dim
    
    # Load input, weight, and bias for this row
    x = tl.load(row_start + tl.arange(0, feature_dim), mask=mask)
    gamma = tl.load(gamma_ptr + tl.arange(0, feature_dim), mask=mask)
    beta = tl.load(beta_ptr + tl.arange(0, feature_dim), mask=mask)
    
    # Compute mean using block reduction
    block_x = tl.zeros([REDUCE_BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, feature_dim, REDUCE_BLOCK_SIZE):
        offsets = i + tl.arange(0, REDUCE_BLOCK_SIZE)
        local_mask = offsets < feature_dim
        values = tl.load(row_start + offsets, mask=local_mask)
        block_x += tl.where(offsets < feature_dim, values, 0.0)
    
    # Sum across blocks and compute mean
    total_sum = tl.sum(block_x, axis=0)
    mean = total_sum / feature_dim
    
    # Compute variance
    block_var = tl.zeros([REDUCE_BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, feature_dim, REDUCE_BLOCK_SIZE):
        offsets = i + tl.arange(0, REDUCE_BLOCK_SIZE)
        local_mask = offsets < feature_dim
        values = tl.load(row_start + offsets, mask=local_mask)
        centered = values - mean
        block_var += tl.where(offsets < feature_dim, centered * centered, 0.0)
    
    # Sum across blocks and compute variance
    total_var = tl.sum(block_var, axis=0)
    var = total_var / feature_dim
    
    # Normalize and apply affine transformation
    std = tl.sqrt(var + eps)
    normalized = (x - mean) / std
    result = normalized * gamma + beta
    
    # Store result
    out_start = out_ptr + row_offset
    tl.store(out_start + tl.arange(0, feature_dim), result, mask=mask)

@triton.jit
def layer_norm_kernel_optimized(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_elements,
    feature_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layer normalization kernel using warp-level primitives"""
    # Each program handles one row
    pid = tl.program_id(0)
    row_offset = pid * feature_dim
    
    # Load row data
    row_start = x_ptr + row_offset
    mask = tl.arange(0, feature_dim) < feature_dim
    
    x = tl.load(row_start + tl.arange(0, feature_dim), mask=mask)
    gamma = tl.load(gamma_ptr + tl.arange(0, feature_dim), mask=mask)
    beta = tl.load(beta_ptr + tl.arange(0, feature_dim), mask=mask)
    
    # Use warp-level primitives for mean computation
    block_size = min(feature_dim, 32)  # Warp size
    warps = (feature_dim + block_size - 1) // block_size
    
    # Compute mean per warp
    warp_sums = tl.zeros([warps], dtype=tl.float32)
    for i in range(warps):
        warp_start = i * block_size
        warp_end = min((i + 1) * block_size, feature_dim)
        warp_mask = tl.arange(warp_start, warp_end) < feature_dim
        
        warp_values = tl.load(row_start + tl.arange(warp_start, warp_end), mask=warp_mask)
        warp_sum = tl.sum(warp_values)
        warp_sums[i] = warp_sum
    
    # Sum all warp sums across warps
    mean = tl.sum(warp_sums) / feature_dim
    
    # Compute variance per warp
    warp_vars = tl.zeros([warps], dtype=tl.float32)
    for i in range(warps):
        warp_start = i * block_size
        warp_end = min((i + 1) * block_size, feature_dim)
        warp_mask = tl.arange(warp_start, warp_end) < feature_dim
        
        warp_values = tl.load(row_start + tl.arange(warp_start, warp_end), mask=warp_mask)
        centered = warp_values - mean
        warp_var = tl.sum(centered * centered)
        warp_vars[i] = warp_var
    
    # Sum all warp vars across warps
    var = tl.sum(warp_vars) / feature_dim
    
    # Normalize and apply affine transformation
    std = tl.sqrt(var + eps)
    normalized = (x - mean) / std
    result = normalized * gamma + beta
    
    # Store result
    out_start = out_ptr + row_offset
    tl.store(out_start + tl.arange(0, feature_dim), result, mask=mask)

@torch.fx.wrap
def triton_layer_norm(x, weight, bias, normalized_shape, eps=1e-5):
    """Wrapper function for optimized layer normalization"""
    # For layer norm, we normalize over the last dimension
    feature_dim = weight.numel()
    
    # Handle different input shapes
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        n_rows = batch_size * seq_len
    else:
        n_rows = 1
        x = x.reshape(-1, feature_dim)
    
    total_elements = x.numel()
    
    # Configure kernel parameters
    BLOCK_SIZE = 32  # Warp size
    REDUCE_BLOCK_SIZE = 128
    
    # Choose kernel based on feature size
    if feature_dim <= 256:
        kernel = layer_norm_kernel_optimized
        reduce_block_size = 32
    else:
        kernel = layer_norm_kernel
        reduce_block_size = REDUCE_BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch Triton kernel
    kernel[(n_rows, 1)](
        x_ptr=x,
        gamma_ptr=weight,
        beta_ptr=bias,
        out_ptr=out,
        n_elements=total_elements,
        feature_dim=feature_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCE_BLOCK_SIZE=reduce_block_size
    )
    
    # Restore original shape if needed
    if x.dim() == 3:
        return out.reshape(batch_size, seq_len, feature_dim)
    else:
        return out.reshape(x.shape)

def pattern(x, normalized_shape, weight, bias, eps):
    """Pattern to match layer normalization operation"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for the replacement function"""
    return (x, weight, bias, normalized_shape, eps)

def replacement_func():
    """Return the optimized layer normalization function"""
    return triton_layer_norm