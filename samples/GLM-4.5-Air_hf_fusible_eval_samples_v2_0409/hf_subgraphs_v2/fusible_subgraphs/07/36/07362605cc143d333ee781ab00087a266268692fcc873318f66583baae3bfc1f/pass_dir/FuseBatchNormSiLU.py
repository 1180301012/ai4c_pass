import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """Match batch norm followed by SiLU activation"""
    batch_norm_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_out = torch.nn.functional.silu(batch_norm_out, inplace=True)
    return silu_out

def replacement_args(x, running_mean, running_var, weight, bias):
    return x, running_mean, running_var, weight, bias

@triton.jit
def fused_batch_norm_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    eps: float,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused BatchNorm + SiLU kernel
    """
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, N)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, C)
    
    # Only iterate if within bounds
    if m_start >= N or n_start >= C:
        return
    
    # Load parameters for this channel
    running_mean_val = tl.load(running_mean_ptr + n_start)
    running_var_val = tl.load(running_var_ptr + n_start)
    weight_val = tl.load(weight_ptr + n_start)
    bias_val = tl.load(bias_ptr + n_start)
    
    # Compute scale and sqrt_inv_std
    sqrt_inv_std = 1.0 / tl.sqrt(running_var_val + eps)
    scale = weight_val * sqrt_inv_std
    
    # Iterate over spatial dimensions
    for h in range(H):
        for w_idx in range(0, W, 4):  # Process 4 elements at a time for vectorization
            # Compute final width index
            w_end = min(w_idx + 4, W)
            
            # Compute offset in input tensor
            offset = (m_start * C * H * W) + (n_start * H * W) + (h * W) + w_idx
            
            # Load input values with masking
            mask = (w_idx + tl.arange(0, 4)) < W
            x_vals = tl.load(x_ptr + offset + tl.arange(0, 4), mask=mask, other=0.0)
            
            # Apply BatchNorm: (x - running_mean) * scale + bias
            batch_norm_vals = (x_vals - running_mean_val) * scale + bias_val
            
            # Apply SiLU: x * sigmoid(x)
            # Use fast sigmoid approximation: 1 / (1 + exp(-x))
            sigmoid_vals = 1.0 / (1.0 + tl.exp(-batch_norm_vals))
            silu_vals = batch_norm_vals * sigmoid_vals
            
            # Store results
            tl.store(out_ptr + offset + tl.arange(0, 4), silu_vals, mask=mask)

@torch.fx.wrap
def fused_batch_norm_silu(x, running_mean, running_var, weight, bias):
    """
    Wrapper function for fused BatchNorm + SiLU
    """
    # Get input dimensions
    N, C, H, W = x.shape  # N is 1 for this pattern
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Configure block sizes
    BLOCK_SIZE_M = 1  # Process one batch at a time (N=1)
    BLOCK_SIZE_N = 256  # Number of channels per block (tunable)
    
    # Calculate grid size
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (C + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_batch_norm_silu_kernel[(
        grid_m,
        grid_n,
    )](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_batch_norm_silu