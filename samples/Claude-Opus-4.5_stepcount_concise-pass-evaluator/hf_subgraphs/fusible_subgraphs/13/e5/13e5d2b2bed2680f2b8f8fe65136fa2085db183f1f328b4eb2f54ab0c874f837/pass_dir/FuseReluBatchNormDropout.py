import torch
import triton
import triton.language as tl


# Pattern matching function - must exactly mirror model.py operations
def pattern(running_mean, running_var, bias, weight, x):
    """
    Match the pattern: relu -> batch_norm (eval mode) -> dropout (p=0.0)
    """
    relu_out = torch.nn.functional.relu(x, inplace=False)
    bn_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    out = torch.nn.functional.dropout(bn_out, p=0.0, training=False)
    return out


# Argument extraction function
def replacement_args(running_mean, running_var, bias, weight, x):
    return (running_mean, running_var, bias, weight, x)


# Triton kernel for fused relu + batch_norm (eval mode) + dropout (no-op)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_relu_bn_kernel(
    x_ptr,              # Input tensor pointer
    running_mean_ptr,   # Running mean pointer
    running_var_ptr,    # Running var pointer
    weight_ptr,         # BN weight pointer
    bias_ptr,           # BN bias pointer
    out_ptr,            # Output tensor pointer
    M,                  # Number of rows (batch dimension)
    N,                  # Number of columns (feature dimension)
    eps: tl.constexpr,  # Epsilon for numerical stability
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for: relu -> batch_norm (eval mode)
    In eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load running mean and var for this feature block
    mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(running_var_ptr + offs_n, mask=mask_n, other=1.0)
    gamma = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    beta = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load input values
    x_offs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    
    # Apply ReLU
    x = tl.maximum(x, 0.0)
    
    # Apply batch norm: y = (x - mean) * inv_std * gamma + beta
    y = (x - mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]
    
    # Store output (dropout with p=0.0 is identity)
    tl.store(out_ptr + x_offs, y, mask=mask)


@torch.fx.wrap
def fused_relu_bn_dropout(running_mean, running_var, bias, weight, x):
    """
    Wrapper function that launches the fused Triton kernel
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    M, N = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Grid dimensions
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
    
    # Launch kernel
    fused_relu_bn_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        M,
        N,
        eps=1e-05,
    )
    
    return out


# Replacement function - returns the wrapper function reference
def replacement_func():
    return fused_relu_bn_dropout