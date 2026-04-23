import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, eps)

# Argument extraction function
def replacement_args(x, weight, bias, eps):
    return (x, weight, bias, eps)

# Triton kernel for LayerNorm
BLOCK_SIZE = 128

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_batch, n_seq, n_hidden,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    start = batch_idx * n_seq * n_hidden + seq_idx * n_hidden
    
    sum_val = 0.0
    sum_sq = 0.0
    
    # Compute mean and variance in first pass
    for i in range(0, n_hidden, BLOCK_SIZE):
        block_mask = i + tl.arange(0, BLOCK_SIZE) < n_hidden
        offsets = start + i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, mask=block_mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)
    
    mean = sum_val / n_hidden
    var = (sum_sq - (sum_val * sum_val) / n_hidden) / n_hidden + eps
    rsqrt_var = tl.rsqrt(var)

    # Normalize and apply scale/bias
    for i in range(0, n_hidden, BLOCK_SIZE):
        block_mask = i + tl.arange(0, BLOCK_SIZE) < n_hidden
        offsets = start + i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, mask=block_mask, other=0.0)
        
        # Load weight and bias
        w = tl.load(weight_ptr + i + tl.arange(0, BLOCK_SIZE), mask=block_mask, other=1.0)
        b = tl.load(bias_ptr + i + tl.arange(0, BLOCK_SIZE), mask=block_mask, other=0.0)
        
        x_norm = (x - mean) * rsqrt_var
        out = x_norm * w + b
        tl.store(out_ptr + offsets, out, mask=block_mask)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps):
    n_batch, n_seq, n_hidden = x.shape
    grid = (n_batch, n_seq)
    out = torch.empty_like(x)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        n_batch, n_seq, n_hidden,
        eps,
        BLOCK_SIZE
    )
    return out

# Replacement function
def replacement_func():
    return layer_norm_wrapper