import torch
import triton
import triton.language as tl

def pattern(tmp_9, in_1, in_0):
    """
    Pattern to match the layer normalization operation:
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    """
    result = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return result

def replacement_args(tmp_9, in_1, in_0):
    return (tmp_9, in_1, in_0)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, 
    hidden_size, seq_len, eps
):
    """Optimized layer normalization kernel using Triton"""
    feature_idx = tl.program_id(0)
    
    # Compute mean for this feature across sequence dimension
    feature_sum = 0.0
    valid_count = 0
    for pos in range(seq_len):
        offset = feature_idx * seq_len + pos
        mask = pos < seq_len
        if mask:
            val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            feature_sum += val
            valid_count += 1
    
    # Use actual count to avoid division by zero
    actual_count = max(valid_count, 1)
    feature_mean = feature_sum / actual_count
    
    # Compute variance for this feature across sequence dimension
    feature_var_sum = 0.0
    for pos in range(seq_len):
        offset = feature_idx * seq_len + pos
        mask = pos < seq_len
        if mask:
            val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            val_centered = val - feature_mean
            feature_var_sum += val_centered * val_centered
    
    feature_var = feature_var_sum / actual_count + eps
    inv_var = 1.0 / tl.sqrt(feature_var)
    
    # Load weight and bias for this feature
    weight = tl.load(weight_ptr + feature_idx)
    bias = tl.load(bias_ptr + feature_idx)
    
    # Apply normalization for all positions of this feature
    for pos in range(seq_len):
        offset = feature_idx * seq_len + pos
        mask = pos < seq_len
        if mask:
            # Load input value
            x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            
            # Normalize and apply weights
            x_centered = x_val - feature_mean
            x_normalized = x_centered * inv_var
            out_val = (x_normalized * weight) + bias
            
            # Store result
            tl.store(out_ptr + offset, out_val, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    """Optimized layer normalization function"""
    # Input x should be [1, hidden_size, seq_len] after transpose (1, 768, 124)
    hidden_size = weight.shape[0]  # 768 from weight_meta.py
    seq_len = x.shape[-1]  # 124 after slicing
    batch_size = x.shape[0]  # 1
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Get eps value
    eps = 1e-05
    
    # Launch kernel - one program per feature
    grid = (hidden_size,)
    
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        hidden_size=hidden_size,
        seq_len=seq_len,
        eps=eps
    )
    
    return out

def replacement_func():
    return optimized_layer_norm