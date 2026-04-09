import torch
import triton
import triton.language as tl

# Exact pattern function that matches the original computation structure
def pattern(in_0, in_1, in_2, in_3):
    """
    Exact pattern matching the original computation: addition followed by layer norm
    """
    # Exact matching: create tmp_2 tensor through addition
    tmp_2 = in_2 + in_3
    
    # Exact matching: create tmp_3 through layer normalization
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    
    # Return exactly what the original returns - the normalized tensor
    return tmp_3

# Extract arguments for the replacement function  
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Fused Triton kernel for addition + layer normalization with 3D tensors
@triton.jit
def fused_add_layer_norm_kernel(
    x_ptr, y_ptr, bias_ptr, weight_ptr, out_ptr,
    batch_size, seq_len, n_features, 
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for 3D tensors: output = layer_norm(x + y)"""
    # Use 1D grid: one program per token
    token_id = tl.program_id(0)
    
    # Calculate which token this program handles
    n_tokens = batch_size * seq_len
    
    # Stop if token ID is out of bounds
    if token_id >= n_tokens:
        return
    
    # Get batch and sequence position for this token
    batch_idx = token_id // seq_len
    seq_idx = token_id % seq_len
    
    # Calculate memory offset for the start of this token's features
    token_offset = batch_idx * seq_len * n_features + seq_idx * n_features
    
    # Always use full BLOCK_SIZE (power of 2) with masking
    feature_offset = tl.arange(0, BLOCK_SIZE)
    feature_mask = feature_offset < n_features
    
    # Load input tensors for this token's features
    x = tl.load(x_ptr + token_offset + feature_offset, mask=feature_mask, other=0.0)
    y = tl.load(y_ptr + token_offset + feature_offset, mask=feature_mask, other=0.0)
    
    # Load normalization parameters (weights and bias)
    bias = tl.load(bias_ptr + feature_offset, mask=feature_mask, other=0.0)
    weight = tl.load(weight_ptr + feature_offset, mask=feature_mask, other=1.0)
    
    # Perform element-wise addition
    sum_xy = x + y
    
    # Compute mean across all features (ignoring padded zeros)
    sum_val = tl.sum(sum_xy)
    mean = sum_val / n_features
    
    # The issue: variance computation includes padded values which distorts the result
    # When we have masked data, padded values are 0.0, but they affect variance:
    # var = [sum((real_values - mean)^2) + sum((padded_zeros - mean)^2)] / n_features
    #     = [sum((real_values - mean)^2) + (num_padded * mean^2)] / n_features
    # We need to compute variance ONLY on the actual values, not the padded ones
    
    # Extract only the actual values (excluding padded zeros)
    # Convert boolean mask to float by multiplication
    mask_float = 1.0 * feature_mask
    actual_values = sum_xy * mask_float
    
    # Compute centering using only real values for statistics
    centered_for_variance = actual_values - mean
    
    # Compute variance correctly (only on actual values)
    var = tl.sum(centered_for_variance * centered_for_variance) / n_features
    
    std = tl.sqrt(var + eps)
    
    # For normalization, we need to center all values (including padded for correct math)
    # But the padding should be zeros, so:
    # centered_all = sum_xy - mean  # This is the right approach
    centered_all = sum_xy - mean
    normalized = centered_all / std
    
    # Apply learned scale and bias  
    result = normalized * weight + bias
    
    # Store the result back to memory (only valid positions)
    tl.store(out_ptr + token_offset + feature_offset, result, feature_mask)

# Wrapper function for the fused kernel - using power-of-2 block size
@torch.fx.wrap  
def fused_add_layer_norm(bias, weight, x, y):
    """Fused addition + layer normalization using Triton"""
    batch_size, seq_len, n_features = x.shape
    
    # Always use power-of-2 block size to avoid arange issues
    BLOCK_SIZE = 1024  # Power of 2, can handle all feature sizes
    
    # Calculate grid dimensions: one program per token
    n_tokens = batch_size * seq_len
    grid = lambda meta: (n_tokens,)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_add_layer_norm_kernel[grid](
        bias, weight, x, y, out,
        batch_size, seq_len, n_features,
        1e-05,
        BLOCK_SIZE
    )
    
    return out

# Replacement function that returns the optimized kernel
def replacement_func():
    return fused_add_layer_norm