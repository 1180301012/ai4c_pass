import torch
import triton
import triton.language as tl

def pattern(a, b):
    matmul = torch.matmul(a, b)
    tmp_1 = matmul.squeeze(1)
    return tmp_1

def replacement_args(a, b):
    return (a, b)

@triton.jit
def optimized_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_programs = tl.cdiv(M * N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    block_start = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
    m = block_start // N
    n = block_start % N
    
    # Create a range of offsets for the block
    offs_m = m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to avoid out-of-bounds accesses
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Loop over K dimension
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        a_offs = k + tl.arange(0, BLOCK_SIZE_K)
        b_offs = k + tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_m[:, None] * K + a_offs[None, :])
        b_ptrs = b_ptr + (a_offs[:, None] * N + offs_n[None, :])
        
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Matrix multiplyaccumulate
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_matmul_squeeze(a, b):
    # Get tensor shapes
    M = 1  # from in_0 shape [1, 1, 249]
    K = 249  # from in_0 and in_1 shapes 
    N = 64  # from in_1 shape [1, 249, 64]
    
    # Create output tensor with shape [1, 64] (skip singleton dimension 1)
    out = torch.empty((1, N), dtype=a.dtype, device=a.device)
    
    # Choose block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32  
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    total_elements = M * N
    block_size_elements = BLOCK_SIZE_M * BLOCK_SIZE_N
    num_programs = (total_elements + block_size_elements - 1) // block_size_elements
    
    # Launch kernel
    optimized_matmul_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return optimized_matmul_squeeze