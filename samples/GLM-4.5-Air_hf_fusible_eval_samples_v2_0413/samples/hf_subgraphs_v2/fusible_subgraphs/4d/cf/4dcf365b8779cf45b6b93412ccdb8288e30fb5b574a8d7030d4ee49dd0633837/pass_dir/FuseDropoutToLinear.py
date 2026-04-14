import torch
import triton
import triton.language as tl
import math

# Pattern 3: dropout + to(dtype) + linear (torchgeometric case)
def pattern(x, weight, bias):
    dropout_x = torch.nn.functional.dropout(x, p=0.0, training=False)
    to = dropout_x.to(torch.float16)  # Will be determined at runtime
    linear = torch.nn.functional.linear(to, weight, bias)
    return linear

# Extract arguments for the replacement
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized kernel: fused dropout + type conversion + linear with proper shape handling
@triton.jit
def fused_dt_to_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M_total,
    K,
    N,
    target_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Scale factor dropout (p=0.0 means identity)
    dropout_scale = 1.0 - 0.0  # 1.0
    
    # Get program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_total, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute memory addresses
    # The first dimension of x might be batch_size * sequence_length
    local_M = pid_m * BLOCK_SIZE_M
    x_mask = local_M < M_total
    
    # For weight and output: [K, N]
    weight_row = tl.arange(0, BLOCK_SIZE_K)
    weight_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    weight_mask = (weight_row < K) & (weight_col < N)

    bias_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_col < N

    # Load x: treat as [M_total, K] by flattening if necessary
    x_ptrs = x_ptr + (local_M * K)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0) * dropout_scale

    # Load weight
    weight_ptrs = weight_ptr + (weight_row[:, None] * N + weight_col[None, :])
    weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    # Load bias
    bias_ptrs = bias_ptr + bias_col
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    # Compute matrix multiplication
    acc = tl.dot(x, weight)

    # Add bias
    acc = acc + bias

    # Store result with dtype conversion
    offset = local_M
    out_ptrs = out_ptr + offset * N + bias_col
    out_val = acc
    
    # Apply dtype conversion
    if target_dtype == 2:  # bfloat16
        out_val = out_val.to(tl.bfloat16)
    else:  # float16
        out_val = out_val.to(tl.float16)
    
    tl.store(out_ptrs, out_val, mask=x_mask & bias_mask)

# Wrapper function for 2D inputs
@torch.fx.wrap
def fused_dt_to_linear(x, weight, bias):
    # Handle 2D input: [M, K]
    M, K = x.shape
    N = bias.shape[0]
    
    # Determine target dtype
    target_dtype = 2 if bias.dtype == torch.bfloat16 else 1
    
    # Simple kernel launch configuration
    n_blocks = (M + 127) // 128  # Use 128 as block size for M dimension
    
    out = torch.empty((M, N), dtype=bias.dtype, device=x.device)
    
    fused_dt_to_linear_kernel[(n_blocks,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        target_dtype=target_dtype,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return out

def replacement_func():
    return fused_dt_to_linear