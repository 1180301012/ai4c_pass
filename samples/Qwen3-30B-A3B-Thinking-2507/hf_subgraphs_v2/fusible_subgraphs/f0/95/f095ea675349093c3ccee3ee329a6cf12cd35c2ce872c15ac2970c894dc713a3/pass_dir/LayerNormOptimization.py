import torch
import triton
import triton.language as tl

# Pattern matching function (must exactly match model.py)
def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    seq_len,
    features,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each block handles one sequence position (across batch & seq_len)
    seq_idx = tl.program_id(0)
    batch_idx = seq_idx // seq_len
    seq_idx = seq_idx % seq_len
    
    # Calculate offset for current sequence position
    start_idx = (batch_idx * seq_len + seq_idx) * features
    
    # Initialize reduction values
    sum = tl.zeros((1,), dtype=tl.float32)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    
    # Load all feature values and compute sum/sum_sq
    for j in range(0, features, BLOCK_SIZE):
        offsets = start_idx + j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (batch_idx * seq_len * features + features)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Calculate mean and variance
    n = tl.cast(features, tl.float32)
    mean = sum / n
    var = (sum_sq / n) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Apply normalization
    for j in range(0, features, BLOCK_SIZE):
        offsets = start_idx + j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (batch_idx * seq_len * features + features)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize and scale
        normalized = (x - mean) * inv_std
        out = normalized * weight + bias
        
        tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, normalized_shape, weight, bias, eps):
    batch, seq_len, features = x.shape
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256  # Optimized for 1024 feature dimension
    grid = batch * seq_len
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_kernel[(grid,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch=batch,
        seq_len=seq_len,
        features=features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_layer_norm