import torch
import triton
import triton.language as tl

def pattern(tmp_12, ln_weight, ln_bias):
    # Pattern matching layer normalization operation
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (tmp_12.shape[-1],), ln_weight, ln_bias, 1e-12)
    return tmp_13

def replacement_args(tmp_12, ln_weight, ln_bias):
    return (tmp_12, ln_weight, ln_bias)

@triton.jit
def layer_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    n_cols, n_rows,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which row this program handles
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Load one row of the input
    row_offset = row_idx * n_cols
    x = tl.load(x_ptr + row_offset, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute mean: μ = (1/n) * Σ(x_i)
    row_sum = tl.sum(x)
    row_mean = row_sum / n_cols
    
    # Compute variance: σ² = (1/n) * Σ((x_i - μ)²)
    x_centered = x - row_mean
    x_squared = x_centered * x_centered
    row_var = tl.sum(x_squared) / n_cols
    
    # Compute standard deviation with epsilon
    row_std = tl.sqrt(row_var + eps)
    
    # Normalize: y_i = (x_i - μ) / σ
    y_normalized = x_centered / row_std
    
    # Apply scale and shift: y_i = γ * y_i + β
    gamma = tl.load(gamma_ptr, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    beta = tl.load(beta_ptr, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    y = y_normalized * gamma + beta
    
    # Store result
    tl.store(y_ptr + row_offset, y, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

@triton.jit
def layer_norm_kernel_vectorized(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    n_cols, n_rows,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which row this program handles
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Load one row of the input with vectorized loads (assuming BLOCK_SIZE is multiple of 2)
    row_offset = row_idx * n_cols
    x = tl.load(x_ptr + row_offset, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute mean using vectorized operations
    row_sum = tl.sum(x)
    row_mean = row_sum / n_cols
    
    # Compute variance using vectorized operations
    x_centered = x - row_mean
    x_squared = x_centered * x_centered
    row_var = tl.sum(x_squared) / n_cols
    
    # Compute standard deviation with epsilon
    row_std = tl.sqrt(row_var + eps)
    
    # Normalize using vectorized operations
    y_normalized = x_centered / row_std
    
    # Apply scale and shift using vectorized loads
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    y = y_normalized * gamma + beta
    
    # Store result
    tl.store(y_ptr + row_offset, y, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-12):
    # Get tensor properties
    n_rows, n_cols = x.shape
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Choose block size based on hidden size
    if n_cols <= 256:
        BLOCK_SIZE = 256
        kernel = layer_norm_kernel
    else:
        BLOCK_SIZE = 512
        kernel = layer_norm_kernel_vectorized
    
    # Calculate number of programs (rows)
    num_programs = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    kernel[(num_programs,)](
        x_ptr=x,
        gamma_ptr=weight,
        beta_ptr=bias,
        y_ptr=y,
        n_cols=n_cols,
        n_rows=n_rows,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

def replacement_func():
    return optimized_layer_norm