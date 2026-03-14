import torch
import triton
import triton.language as tl


def pattern(x, normalized_shape, weight, bias):
    """Pattern: just layernorm"""
    out = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, 1e-05)
    return out


def replacement_args(x, normalized_shape, weight, bias):
    return (x, normalized_shape, weight, bias)


@triton.jit
def layernorm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layernorm kernel with better memory coalescing"""
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Calculate offsets
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input with coalesced access
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Compute mean - only include valid elements
    x_masked = tl.where(mask, x, 0.0)
    x_sum = tl.sum(x_masked, axis=0)
    mean = x_sum / n_cols
    
    # Center the data
    centered = x - mean
    centered_masked = tl.where(mask, centered, 0.0)
    
    # Compute variance - only include valid elements
    var_sum = tl.sum(centered_masked * centered_masked, axis=0)
    var = var_sum / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias (these are small and can be cached)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Normalize and scale
    out = centered * rstd * weight + bias
    
    # Store result with coalesced access
    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


@torch.fx.wrap
def layernorm_impl(x, normalized_shape, weight, bias):
    """Optimized layernorm implementation"""
    # Get dimensions
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_dim = x.shape[2]
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel with optimal block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    n_rows = batch_size * seq_len
    grid = (n_rows,)
    
    layernorm_kernel_optimized[grid](
        x,
        weight,
        bias,
        out,
        n_rows,
        hidden_dim,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return layernorm_impl