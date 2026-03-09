import torch
import triton
import triton.language as tl

# Pattern matching function for matmul operation
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    # Include other operations but they're handled by other passes
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 1, 1, 1)  # dummy reshape for pattern matching
    tmp_5 = torch.functional.split(tmp_4, [1, 1, 1], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1] 
    tmp_8 = tmp_5[2]
    return tmp_0, tmp_6, tmp_7, tmp_8, tmp_1

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
    # Determine program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_id = pid // num_pid_n
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    # Create pointers for blocks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load block from A
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M)[:, None] & (offs_k[None, :] < K), other=0.0)
        
        # Load block from B  
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K)[:, None] & (offs_bn[None, :] < N), other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M)[:, None] & (offs_cn[None, :] < N))

# Kernel wrapper
@torch.fx.wrap
def optimized_matmul(in_1, in_0):
    # Get input shapes
    batch, heads, seq_len, dim = in_1.shape
    _, _, _, dim2 = in_0.shape
    
    # Output shape
    out_shape = (batch, heads, seq_len, dim2)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Set grid size and block size
    M = seq_len
    N = dim2 
    K = dim
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Launch kernel
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

# Replacement function
def replacement_func():
    return optimized_matmul

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_1, in_0)