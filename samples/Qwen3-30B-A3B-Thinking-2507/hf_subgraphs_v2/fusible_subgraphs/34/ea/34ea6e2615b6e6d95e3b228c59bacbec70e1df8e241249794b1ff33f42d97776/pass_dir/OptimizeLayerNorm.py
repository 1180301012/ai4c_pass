import torch
import triton
import triton.language as tl

# Pattern matching function
# Match the layer_norm operation specifically as it appears in the model
def pattern(x, weight, bias, eps):
    # Note: Exact matching of the operation in model.py (including positional args)
    ln_out = torch.nn.functional.layer_norm(x, (768,), weight, bias, eps)
    return x, ln_out  # Must return all observable outputs (x is tmp_7, which is part of the model's return)

def replacement_args(x, weight, bias, eps):
    # Extract all needed arguments for the optimized kernel
    return (x, weight, bias, eps)

# Triton kernel for optimized layer normalization
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    n_seq, n_features, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Process one sequence position per block
    seq_id = tl.program_id(0)
    start_idx = seq_id * n_features

    # Load the entire sequence's features (768 features)
    x = tl.load(x_ptr + start_idx, n_features)
    
    # Compute mean and variance over features
    sum_x = tl.sum(x)
    sum_sq = tl.sum(x * x)
    mean = sum_x / n_features
    var = (sum_sq / n_features) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Load weight/bias vectors
    weight = tl.load(weight_ptr, n_features)
    bias = tl.load(bias_ptr, n_features)

    # Apply layer norm
    y = (x - mean) * inv_std * weight + bias
    tl.store(y_ptr + start_idx, y)


# Wrapper for Triton kernel
@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps):
    # Determine sequence length (256) and feature length (768)
    n_seq = x.shape[1]  # Shape is [1, 256, 768]
    n_features = x.shape[2]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with block size = 768 (feature dimension)
    num_blocks = n_seq
    BLOCK_SIZE = n_features
    
    layer_norm_kernel[(num_blocks,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        y_ptr=out,
        n_seq=n_seq,
        n_features=n_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return optimized_layer_norm