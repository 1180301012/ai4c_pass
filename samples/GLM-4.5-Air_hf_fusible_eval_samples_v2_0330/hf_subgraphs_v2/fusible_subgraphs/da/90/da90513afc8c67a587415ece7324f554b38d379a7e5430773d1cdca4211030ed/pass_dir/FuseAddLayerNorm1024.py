import torch
import triton
import triton.language as tl
import math

# Pattern matching function - 1024 feature variant
def pattern(in_0, in_1, in_2, in_3):
    """
    Matches the computation pattern for models with 1024 features: addition followed by layer normalization
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused add + layer normalization
@triton.jit
def fused_add_layer_norm_kernel(
    bias_ptr,
    weight_ptr,
    x1_ptr,
    x2_ptr,
    out_ptr,
    feature_size,
    batch_size,
    seq_len,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch sequence
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Load bias and weight for this feature position
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size
    
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    
    # Load input tensors with contiguous memory access
    x1_base = batch_idx * seq_len * feature_size + seq_idx * feature_size
    x2_base = batch_idx * seq_len * feature_size + seq_idx * feature_size
    
    x1_vals = tl.load(x1_ptr + x1_base + offsets, mask=mask, other=0.0)
    x2_vals = tl.load(x2_ptr + x2_base + offsets, mask=mask, other=0.0)
    
    # Perform fused addition
    added_vals = x1_vals + x2_vals
    
    # Compute mean and variance for this sequence position
    sum_val = tl.sum(added_vals)
    mean = sum_val / feature_size
    
    # Compute variance using standard layer normalization formula
    normalized_vals = added_vals - mean
    normalized_sq = normalized_vals * normalized_vals
    var_mean = tl.sum(normalized_sq) / feature_size
    
    # Standard layer normalization with better numerical stability
    denominator = tl.sqrt(var_mean + eps)
    # Handle case where variance is close to zero
    safe_denominator = tl.where(denominator < eps, eps, denominator)
    normalized_vals = normalized_vals / safe_denominator * weight + bias
    
    # Store result with proper memory alignment
    out_base = batch_idx * seq_len * feature_size + seq_idx * feature_size
    tl.store(out_ptr + out_base + offsets, normalized_vals, mask=mask)

# Optimized kernel wrapper with smart block sizing
@torch.fx.wrap
def fused_add_layer_norm(bias, weight, x1, x2):
    feature_size = bias.shape[0]
    batch_size, seq_len, _ = x1.shape
    total_elements = batch_size * seq_len
    
    # Use power-of-2 block size for better GPU utilization and Triton compatibility
    BLOCK_SIZE = min(1024, triton.next_power_of_2(feature_size))
    
    # Launch grid with one program per batch/sequence position
    grid = (total_elements,)
    
    # Create output tensor
    out = torch.empty_like(x1)
    
    # Launch kernel
    fused_add_layer_norm_kernel[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        x1_ptr=x1,
        x2_ptr=x2,
        out_ptr=out,
        feature_size=feature_size,
        batch_size=batch_size,
        seq_len=seq_len,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_add_layer_norm