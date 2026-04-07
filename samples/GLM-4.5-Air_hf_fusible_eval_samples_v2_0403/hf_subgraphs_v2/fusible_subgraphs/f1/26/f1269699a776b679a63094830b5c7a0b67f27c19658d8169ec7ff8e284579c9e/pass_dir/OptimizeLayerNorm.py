import torch
import triton
import triton.language as tl

def pattern(embedding_add_result, layer_norm_weight, layer_norm_bias, epsilon=1e-05):
    """Pattern: Layer normalization operation
    Matches: torch.nn.functional.layer_norm(embedding_add_result, normalized_shape, layer_norm_weight, layer_norm_bias, epsilon)
    """
    result = torch.nn.functional.layer_norm(embedding_add_result, embedding_add_result.shape[-1:], layer_norm_weight, layer_norm_bias, epsilon)
    return result

def replacement_args(embedding_add_result, layer_norm_weight, layer_norm_bias, epsilon=1e-05):
    """Extract arguments needed for optimized layer norm kernel"""
    return (embedding_add_result, layer_norm_weight, layer_norm_bias, epsilon)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized layer normalization kernel using Triton
    
    Computes: y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta
    where mu is the mean and sigma^2 is the variance along the last dimension
    """
    # Row offsets (for each row in the batch)
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Check bounds
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols
    
    # Load current row
    x_row = tl.load(
        x_ptr + row_idx * n_cols + col_offsets,
        mask=col_mask,
        other=0.0
    )
    
    # Compute mean for this row
    row_mean = tl.sum(x_row, axis=0) / n_cols
    
    # Compute variance for this row
    x_centered = x_row - row_mean
    x_squared = x_centered * x_centered
    row_var = tl.sum(x_squared, axis=0) / n_cols
    
    # Compute inverse std deviation
    row_inv_std = tl.rsqrt(row_var + eps)
    
    # Apply layer normalization
    y_row = x_centered * row_inv_std
    
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + col_offsets, mask=col_mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=col_mask, other=0.0)
    
    # Scale and shift
    y_row = y_row * gamma + beta
    
    # Store result
    tl.store(
        y_ptr + row_idx * n_cols + col_offsets,
        y_row,
        mask=col_mask
    )

@triton.jit
def layer_norm_kernel_warp_optimized(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Warp-optimized layer normalization kernel for better performance"""
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    m_end = min(m_start + BLOCK_SIZE_M, n_rows)
    n_end = min(n_start + BLOCK_SIZE_N, n_cols)
    
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < n_rows
    n_mask = n_offsets < n_cols
    
    # Process each row in the block
    for row_idx, m_mask_val in zip(m_offsets, m_mask):
        if not m_mask_val:
            continue
            
        # Load current row
        x_row = tl.load(
            x_ptr + row_idx * n_cols + n_offsets,
            mask=n_mask,
            other=0.0
        )
        
        # Compute mean
        row_mean = tl.sum(x_row, axis=0) / tl.sum(n_mask).to(tl.float32)
        
        # Compute variance
        x_centered = x_row - row_mean
        x_squared = x_centered * x_centered
        row_var = tl.sum(x_squared, axis=0) / tl.sum(n_mask).to(tl.float32)
        
        # Compute inverse std
        row_inv_std = tl.rsqrt(row_var + eps)
        
        # Apply normalization
        y_row = x_centered * row_inv_std
        
        # Load and apply gamma/beta
        gamma = tl.load(gamma_ptr + n_offsets, mask=n_mask, other=1.0)
        beta = tl.load(beta_ptr + n_offsets, mask=n_mask, other=0.0)
        y_row = y_row * gamma + beta
        
        # Store result
        tl.store(
            y_ptr + row_idx * n_cols + n_offsets,
            y_row,
            mask=n_mask
        )

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-05):
    """Optimized layer normalization using Triton"""
    if x.dim() != 3:
        raise ValueError("Input must be 3D tensor (batch, seq_len, hidden_size)")
    
    batch_size, seq_len, hidden_size = x.shape
    
    # Permute to (batch_size * seq_len, hidden_size) for easier processing
    x_flat = x.reshape(-1, hidden_size)
    
    # Allocate output
    y = torch.empty_like(x_flat)
    
    # Choose between simple and optimized kernel based on size
    if hidden_size <= 1024:
        # Use kernel optimized for typical transformer hidden sizes
        BLOCK_SIZE_N = min(256, hidden_size)
        BLOCK_SIZE_M = 16
        
        grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = 1  # Process entire hidden dimension at once
        
        layer_norm_kernel_warp_optimized[(grid_m, grid_n)](
            x_ptr=x_flat,
            gamma_ptr=weight,
            beta_ptr=bias,
            y_ptr=y,
            n_rows=batch_size * seq_len,
            n_cols=hidden_size,
            eps=eps,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    else:
        # Use simpler kernel for large hidden sizes
        BLOCK_SIZE_N = 128
        grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        
        for i in range(0, batch_size * seq_len, BLOCK_SIZE_M):
            end_i = min(i + BLOCK_SIZE_M, batch_size * seq_len)
            layer_norm_kernel[(end_i - i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (
                hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N](
                x_ptr=x_flat + i * hidden_size,
                gamma_ptr=weight,
                beta_ptr=bias,
                y_ptr=y + i * hidden_size,
                n_rows=end_i - i,
                n_cols=hidden_size,
                eps=eps,
                BLOCK_SIZE_M=min(BLOCK_SIZE_M, end_i - i),
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
    
    return y.reshape(x.shape)

def replacement_func():
    """Return the optimized layer normalization function"""
    return optimized_layer_norm