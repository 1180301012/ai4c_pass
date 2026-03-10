import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Pattern: torch.nn.functional.layer_norm(tmp_7, (320,), tmp_2, tmp_1, 1e-05)
    # x is the input to be normalized, weight is the norm weight, bias is the norm bias
    return torch.nn.functional.layer_norm(x, (320,), weight, bias, 1e-05)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    feature_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + (offsets % feature_size), mask=offsets % feature_size < feature_size, other=0.0)
    bias = tl.load(bias_ptr + (offsets % feature_size), mask=offsets % feature_size < feature_size, other=0.0)
    
    # Calculate mean
    block_sum = tl.sum(x, axis=0)
    block_mean = block_sum / feature_size
    
    # Calculate variance
    x_centered = x - block_mean
    x_squared = x_centered * x_centered
    block_sum_sq = tl.sum(x_squared, axis=0)
    block_var = block_sum_sq / feature_size
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(block_var + eps)
    x_normalized = x_centered * inv_std
    
    # Apply weight and bias
    out = x_normalized * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    """
    Optimized layer normalization using Triton kernel
    x: [batch_size, seq_len, feature_size] or [batch_size, feature_size]
    weight: [feature_size] 
    bias: [feature_size]
    """
    # Handle different input shapes
    if x.dim() == 2:
        # [batch_size, feature_size]
        batch_size, feature_size = x.shape
        N = batch_size * feature_size
    elif x.dim() == 3:
        # [batch_size, seq_len, feature_size]
        batch_size, seq_len, feature_size = x.shape
        N = batch_size * seq_len * feature_size
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Set block size and launch grid
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    layernorm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=N,
        feature_size=feature_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm