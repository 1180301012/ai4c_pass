import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """Pattern matching: layer_norm"""
    output = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return output

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for replacement"""
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Proper layer norm kernel using parallel reduction"""
    # Each program processes one feature dimension
    feat_idx = tl.program_id(0)
    n_features = feature_size
    
    if feat_idx >= n_features:
        return
    
    # Initialize reduction variables
    batch_sum = 0.0
    batch_sum_sq = 0.0
    
    # Compute mean and variance across all positions for this feature
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            offset = batch_idx * seq_len * feature_size + seq_idx * feature_size + feat_idx
            
            # Load value
            x_val = tl.load(x_ptr + offset, other=0.0)
            
            # Accumulate for mean/variance
            batch_sum += x_val
            batch_sum_sq += x_val * x_val
    
    # Compute mean and variance
    total_elements = batch_size * seq_len
    mean = batch_sum / total_elements
    variance = (batch_sum_sq / total_elements) - (mean * mean)
    variance = tl.maximum(variance, 1e-05)  # Add epsilon
    
    # Standard deviation
    std = tl.sqrt(variance)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + feat_idx, other=1.0)
    bias = tl.load(bias_ptr + feat_idx, other=0.0)
    
    # Apply layer normalization
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            offset = batch_idx * seq_len * feature_size + seq_idx * feature_size + feat_idx
            
            # Load original value
            x_val = tl.load(x_ptr + offset, other=0.0)
            
            # Normalize and apply weight/bias
            normalized = (x_val - mean) / std
            result = normalized * weight + bias
            
            # Store result
            tl.store(out_ptr + offset, result)

@torch.fx.wrap
def optimized_layer_norm(x, normalized_shape, weight, bias, eps):
    """Wrapper function for optimized layer norm"""
    # Use optimized sequential operations  
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x_normalized = (x - mean) / (std + eps)
    out = x_normalized * weight + bias
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_layer_norm