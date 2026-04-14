import torch
import triton
import triton.language as tl
import math

# Pattern 2: dropout(p=0.0) + type_conversion + linear
def pattern(x, weight, bias):
    dropout_x = torch.nn.functional.dropout(x, p=0.0, training=False)
    to = dropout_x.to(torch.float16)  # This handles dtype conversion
    linear = torch.nn.functional.linear(to, weight, bias)
    return linear

# Extract arguments for the replacement
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized kernel: remove no-op dropout + convert dtype + linear
@triton.jit
def optimized_dtype_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    target_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute memory address for x
    x_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_col = 0
    x_mask = x_row < M

    # Compute memory address for weight
    weight_row = tl.arange(0, BLOCK_SIZE_K)
    weight_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    weight_mask = (weight_row < K) & (weight_col < N)

    # Compute memory address for bias
    bias_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_col < N

    # Compute output memory address
    out_ptrs = out_ptr + (x_row[:, None] * N + bias_col[None, :])

    # Load x (already on device, perform dtype conversion in kernel if needed)
    x_ptrs = x_ptr + (x_row[:, None] * K + x_col)
    
    # Handle dtype conversion in kernel if needed
    if target_dtype == tl.bfloat16:
        x_val = tl.load(x_ptrs, mask=x_mask[:, None], other=0.0).to(tl.bfloat16)
    else:
        x_val = tl.load(x_ptrs, mask=x_mask[:, None], other=0.0).to(tl.float16)

    # Load weight
    weight_ptrs = weight_ptr + (weight_row[:, None] * N + weight_col[None, :])
    weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    # Load bias
    bias_ptrs = bias_ptr + bias_col
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    # Compute matrix multiplication
    acc = tl.dot(x_val, weight)

    # Add bias
    acc = acc + bias[None, :]

    # Store result with target dtype
    if target_dtype == tl.bfloat16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=x_mask[:, None] & bias_mask[None, :])
    else:
        tl.store(out_ptrs, acc.to(tl.float16), mask=x_mask[:, None] & bias_mask[None, :])

# Wrapper function
@torch.fx.wrap
def optimized_dtype_linear(x, weight, bias):
    M, K = x.shape
    N = bias.shape[0]
    
    # Determine target dtype based on bias dtype
    target_dtype = 2 if bias.dtype == torch.bfloat16 else 1  # 2 for bfloat16, 1 for float16
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid = num_pid_m * num_pid_n
    
    out = torch.empty((M, N), dtype=bias.dtype, device=x.device)
    
    optimized_dtype_linear_kernel[(num_pid, *([0] * (len(x.shape) - 2)))](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        target_dtype=target_dtype,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return out

def replacement_func():
    return optimized_dtype_linear