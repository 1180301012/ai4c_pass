import torch
import triton
import triton.language as tl

@triton.jit
def optimized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # Compute memory address of blocK
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) 
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) 
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Write back result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@torch.fx.wrap
def optimized_matmul(a, b):
    # Get tensor shapes correctly
    # a: [batch, heads, M, K] - e.g. [1, 8, 50, 19]
    # b: [batch, heads, K, N] - e.g. [1, 8, 19, 19] 
    # Expected output: [batch, heads, M, N] - e.g. [1, 8, 50, 19]
    batch_size, heads, M, K = a.shape
    _, _, K_, N = b.shape
    
    # Verify K dimensions match
    assert K == K_, f"Dimension mismatch: {K} != {K_}"
    
    # Set block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Compute grid size using regular Python math
    import math
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = num_pid_m * num_pid_n
    num_groups = batch_size * heads  # Each head needs computation
    grid = (num_groups * num_pid_in_group,)
    
    # Create output tensor with correct shape
    output = torch.empty(batch_size, heads, M, N, 
                        dtype=a.dtype, device=a.device)
    
    # Launch kernel with correct strides
    optimized_matmul_kernel[grid](
        a, b, output,
        M, N, K,  # Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,  # Block sizes
        a.stride(1), a.stride(3),  # strides for a: (heads_stride, K_stride)
        b.stride(2), b.stride(3),  # strides for b: (K_stride, N_stride)  
        output.stride(1), output.stride(3)  # strides for output: (heads_stride, N_stride)
    )
    
    return output

def pattern(a, b):
    return a @ b

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    return optimized_matmul