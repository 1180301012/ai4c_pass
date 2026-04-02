import torch
import triton
import triton.language as tl

def pattern(softmax_out, value_layer):
    """
    Match the matmul operation in attention computation
    Pattern: torch.matmul(softmax_out, value_layer)
    Returns the matmul result
    """
    matmul = torch.matmul(softmax_out, value_layer)
    return matmul

def replacement_args(softmax_out, value_layer):
    return (softmax_out, value_layer)

@triton.jit
def optimized_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel using Triton
    Based on Triton tutorial with attention-specific optimizations
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@torch.fx.wrap
def optimized_matmul(a, b):
    """
    Optimized matrix multiplication using Triton kernel
    Handles different tensor shapes efficiently
    """
    # Get tensor shapes and handle broadcasting
    shape_a = a.shape
    shape_b = b.shape
    
    # For attention computation, typical shapes are:
    # a: [batch_size, num_heads, seq_len, seq_len] 
    # b: [batch_size, num_heads, seq_len, head_dim]
    # c: [batch_size, num_heads, seq_len, head_dim]
    
    # Flatten dimensions for matmul while preserving batch and head dimensions
    if len(shape_a) == 4 and len(shape_b) == 4:
        batch, heads, seq_len_a, _ = shape_a
        batch_b, heads_b, seq_len_b, head_dim = shape_b
        
        # Assert compatibility
        assert batch == batch_b and heads == heads_b and seq_len_a == seq_len_b
        
        # Reshape to 2D for each batch and head
        a_flat = a.reshape(batch * heads, seq_len_a, seq_len_a)
        b_flat = b.reshape(batch * heads, seq_len_b, head_dim)
        
        M, K = a_flat.shape[-2], a_flat.shape[-1]
        N = b_flat.shape[-1]
        
        # Determine optimal block sizes
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        # Adjust based on tensor sizes
        if M < 64:
            BLOCK_SIZE_M = 32
        if N < 64:
            BLOCK_SIZE_N = 32
        if K < 32:
            BLOCK_SIZE_K = 16
            
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        GROUP_SIZE_M = 8  # Use smaller group size for better load balancing
        
        grid = (num_pid_m * num_pid_n,)
        
        # Allocate output tensor
        c_flat = torch.empty((batch * heads, seq_len_a, head_dim), dtype=a.dtype, device=a.device)
        
        # Launch kernel
        optimized_matmul_kernel[grid](
            a_ptr=a_flat,
            b_ptr=b_flat,
            c_ptr=c_flat,
            M=M,
            N=N,
            K=K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )
        
        # Reshape back to original format
        return c_flat.reshape(batch, heads, seq_len_a, head_dim)
    
    else:
        # Fallback to regular matmul for unsupported shapes
        return torch.matmul(a, b)

def replacement_func():
    """Return optimized matmul function"""
    return optimized_matmul