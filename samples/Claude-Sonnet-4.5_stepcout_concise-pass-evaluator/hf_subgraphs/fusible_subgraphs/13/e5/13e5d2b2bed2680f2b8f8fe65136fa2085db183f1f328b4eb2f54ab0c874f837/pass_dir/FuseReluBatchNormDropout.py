import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(running_mean, running_var, bias, weight, x):
    """
    Match ReLU + BatchNorm + Dropout pattern
    """
    relu_out = torch.nn.functional.relu(x, inplace=False)
    bn_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    dropout_out = torch.nn.functional.dropout(bn_out, p=0.0, training=False)
    return (dropout_out,)

# Argument extraction function
def replacement_args(running_mean, running_var, bias, weight, x):
    return (running_mean, running_var, bias, weight, x)

# Optimized Triton kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_relu_bn_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load batch norm parameters (broadcasted along feature dimension)
    running_mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    running_var = tl.load(running_var_ptr + offs_n, mask=mask_n, other=0.0)
    weight = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Compute std
    std = tl.sqrt(running_var + eps)
    
    # Load input data and process row by row within the block
    for i in range(BLOCK_SIZE_M):
        m_idx = pid_m * BLOCK_SIZE_M + i
        if m_idx < M:
            # Load input
            x_ptrs = x_ptr + m_idx * N + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=0.0)
            
            # ReLU
            x_relu = tl.maximum(x, 0.0)
            
            # Batch norm: (x - mean) / std * weight + bias
            x_normalized = (x_relu - running_mean) / std
            out = x_normalized * weight + bias
            
            # Store result
            out_ptrs = out_ptr + m_idx * N + offs_n
            tl.store(out_ptrs, out, mask=mask_n)

# Kernel wrapper
@torch.fx.wrap
def fused_relu_bn_dropout(running_mean, running_var, bias, weight, x):
    M, N = x.shape
    eps = 1e-05
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Grid configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    fused_relu_bn_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        M, N, eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_relu_bn_dropout