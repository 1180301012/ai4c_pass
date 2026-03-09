import torch
import triton
import triton.language as tl

# Pattern matching function for just the matmul operation
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    # The matmul result is used as one of the observable outputs
    # We need to include enough context to make the pattern match
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    # The tmp_1 is also observable, so return both
    return tmp_0, tmp_1

# Optimized matmul kernel
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_id = pid // num_pid_n
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M)[:, None] & (offs_k[None, :] < K), other=0.0)
        
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K)[:, None] & (offs_bn[None, :] < N), other=0.0)
        
        accumulator += tl.dot(a, b)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M)[:, None] & (offs_cn[None, :] < N))

# Kernel wrapper for matmul
@torch.fx.wrap
def optimized_matmul(in_1, in_0):
    batch, heads, seq_len, dim = in_1.shape
    _, _, _, dim2 = in_0.shape
    
    out_shape = (batch, heads, seq_len, dim2)
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    M, N, K = seq_len, dim2, dim
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        in_1,
        in_0,
        out,
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )
    
    return out

# Simple slicing operation (already optimized in PyTorch)
def simple_slice(tensor, start_idx=1):
    return tensor[slice(None, None, None), slice(None, None, None), slice(start_idx, None, None), slice(None, None, None)]

# Replacement function that returns observable outputs
def replacement_func():
    def optimized_forward(in_0, in_1, in_2):
        # Step 1: Optimized matrix multiplication
        tmp_0 = optimized_matmul(in_1, in_0)
        
        # Step 2: Original slicing for tmp_1 (this is already efficient)
        tmp_1 = simple_slice(in_1, start_idx=1)
        
        # Return the observable outputs that the pattern matched
        return tmp_0, tmp_1
    
    return optimized_forward

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)