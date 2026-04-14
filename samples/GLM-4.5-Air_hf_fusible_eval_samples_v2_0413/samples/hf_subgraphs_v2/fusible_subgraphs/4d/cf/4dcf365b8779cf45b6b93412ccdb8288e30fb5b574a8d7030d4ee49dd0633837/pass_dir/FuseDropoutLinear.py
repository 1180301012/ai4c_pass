import torch
import triton
import triton.language as tl
import math

# Pattern 1: dropout(p > 0) + linear
def pattern(x, weight, bias):
    dropout_x = torch.nn.functional.dropout(x, 0.1, False, False)
    linear = torch.nn.functional.linear(dropout_x, weight, bias)
    return linear

# Extract arguments for the replacement
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized kernel: fused dropout + linear with dropout scaling
@triton.jit
def fused_dropout_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    dropout_prob: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Scale factor for dropout during inference
    dropout_scale = 1.0 - dropout_prob
    
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
    x_col = 0  # x is [M, K] where M might be batch_size * sequence_length
    x_mask = x_row < M

    # Compute memory address for weight
    weight_row = tl.arange(0, BLOCK_SIZE_K)
    weight_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    weight_mask = (weight_row < K) & (weight_col < N)

    # Compute memory address for bias
    bias_col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_col < N

    # Load x with dropout scaling applied
    x_ptrs = x_ptr + (x_row[:, None] * K + x_col)
    x = tl.load(x_ptrs, mask=x_mask[:, None], other=0.0) * dropout_scale

    # Load weight
    weight_ptrs = weight_ptr + (weight_row[:, None] * N + weight_col[None, :])
    weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    # Load bias
    bias_ptrs = bias_ptr + bias_col
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    # Compute matrix multiplication
    acc = tl.dot(x, weight)

    # Add bias
    acc = acc + bias[None, :]

    # Store result
    out_ptrs = out_ptr + (x_row[:, None] * N + bias_col[None, :])
    tl.store(out_ptrs, acc, mask=x_mask[:, None] & bias_mask[None, :])

# Wrapper function for 2D inputs (torchgeometric case)
@torch.fx.wrap
def fused_dropout_linear(x, weight, bias):
    # Handle 2D input: [M, K]
    M, K = x.shape
    N = bias.shape[0]
    
    # Simple kernel launch configuration
    n_blocks = (M + 127) // 128  # Use 128 as block size for M dimension
    
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    fused_dropout_linear_kernel[(n_blocks,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        dropout_prob=0.1,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return out

def replacement_func():
    return fused_dropout_linear