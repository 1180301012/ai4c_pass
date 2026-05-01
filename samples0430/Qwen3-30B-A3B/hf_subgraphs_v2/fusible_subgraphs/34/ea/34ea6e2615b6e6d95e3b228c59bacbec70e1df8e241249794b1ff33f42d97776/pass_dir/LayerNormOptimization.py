import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, norm_shape, weight, bias, eps):
    # Match exact pattern with normalized_shape (768,) and eps=1e-05
    if norm_shape != (768,) or eps != 1e-05:
        return None
    return torch.nn.functional.layer_norm(x, norm_shape, weight, bias, eps)

# Argument extraction function
def replacement_args(x, norm_shape, weight, bias, eps):
    return (x, weight, bias)

# Triton kernel
@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_seq,
    n_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    # Load input row for current sequence element
    input_row = tl.load(
        input_ptr + block_id * n_features + tl.arange(0, n_features),
        mask=tl.arange(0, n_features) < n_features
    )
    # Load weight and bias for all features
    weight = tl.load(
        weight_ptr + tl.arange(0, n_features),
        mask=tl.arange(0, n_features) < n_features
    )
    bias = tl.load(
        bias_ptr + tl.arange(0, n_features),
        mask=tl.arange(0, n_features) < n_features
    )
    
    # Compute sum and sum of squares for reduction
    sum_val = 0.0
    sumsq_val = 0.0
    for i in range(n_features):
        x_val = input_row[i]
        sum_val += x_val
        sumsq_val += x_val * x_val
    
    mean = sum_val / n_features
    var = sumsq_val / n_features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization
    output_row = (input_row - mean) * inv_std * weight + bias
    
    # Store output
    tl.store(
        output_ptr + block_id * n_features + tl.arange(0, n_features),
        output_row,
        mask=tl.arange(0, n_features) < n_features
    )

# Kernel wrapper
@torch.fx.wrap
def optimized_layernorm(x, weight, bias):
    # Extract tensor properties
    batch_size, seq_len, features = x.shape
    n_seq = seq_len
    n_features = features
    eps = 1e-05
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel: 1 block per sequence element, all features processed in each block
    grid = (n_seq,)
    
    # BLOCK_SIZE = n_features (768) to use full block size
    layernorm_kernel[grid](
        x, weight, bias, out,
        n_seq, n_features,
        eps,
        n_features
    )
    
    return out

# Replacement function

def replacement_func():
    return optimized_layernorm