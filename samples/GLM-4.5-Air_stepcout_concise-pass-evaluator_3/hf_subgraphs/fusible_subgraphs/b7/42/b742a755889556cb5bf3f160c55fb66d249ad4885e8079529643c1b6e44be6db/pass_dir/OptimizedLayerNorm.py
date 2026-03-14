import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Match layer norm operation
    result = torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-05)
    return result

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Custom LayerNorm kernel
@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Offset for current row
    row_offset = row_idx * n_cols
    
    # Load current row
    x = tl.load(x_ptr + row_offset + col_idx, mask=(col_idx < n_cols), other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / n_cols
    std = tl.sqrt(var + eps)
    
    # Normalize
    x_norm = x_centered / std
    
    # Load weight and bias
    weight = tl.load(weight_ptr + col_idx, mask=(col_idx < n_cols), other=0.0)
    bias = tl.load(bias_ptr + col_idx, mask=(col_idx < n_cols), other=0.0)
    
    # Apply linear transformation
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + row_offset + col_idx, out, mask=(col_idx < n_cols))

@torch.fx.wrap
def optimized_layernorm(x, weight, bias):
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    
    # Calculate grid size
    BLOCK_M = 32
    BLOCK_N = 128
    grid_m = (n_rows + BLOCK_M - 1) // BLOCK_M
    grid_n = (n_cols + BLOCK_N - 1) // BLOCK_N
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    layernorm_kernel[grid_m, grid_n](
        x,
        weight,
        bias,
        out,
        n_rows,
        n_cols,
        1e-05,
        BLOCK_M,
        BLOCK_N
    )
    
    return out

def replacement_func():
    return optimized_layernorm