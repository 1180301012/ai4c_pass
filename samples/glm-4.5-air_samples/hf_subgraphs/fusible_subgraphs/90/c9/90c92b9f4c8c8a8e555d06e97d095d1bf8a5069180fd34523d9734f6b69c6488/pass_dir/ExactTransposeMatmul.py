import torch
import triton
import triton.language as tl

# Exact pattern from the problem description example
def pattern(a, b):
    t = a.transpose(-1, -2)
    out = t @ b
    return t, out

def replacement_args(a, b):
    return (a, b)

@triton.jit
def transpose_matmul_kernel(
    a_ptr,
    b_ptr,
    t_ptr,
    out_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication kernel with transpose
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges each program will compute
    range_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    range_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = range_m < m
    mask_n = range_n < n
    
    # Accumulate result in registers
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_idx in range(0, k, BLOCK_SIZE_K):
        range_k = k_idx + tl.arange(0, BLOCK_SIZE_K)
        mask_k = range_k < k
        
        # Load a (transposed on the fly)
        a_offsets = (range_m[:, None] * k + range_k[None, :]).to(tl.int32)
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load b
        b_offsets = (range_k[:, None] * n + range_n[None, :]).to(tl.int32)
        b_vals = tl.load(b_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(a_vals, b_vals)
    
    # Store transpose result
    t_offsets = (range_m[:, None] * k + range_k[None, :]).to(tl.int32)
    tl.store(t_ptr + t_offsets, a_vals, mask=mask_m[:, None] & mask_k[None, :])
    
    # Store output
    out_offsets = (range_m[:, None] * n + range_n[None, :]).to(tl.int32)
    tl.store(out_ptr + out_offsets, accumulator, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def triton_transpose_matmul(a, b):
    m, k = a.shape
    k2, n = b.shape
    
    # Ensure compatibility
    if k != k2:
        raise ValueError(f"Inner dimensions must match: got {k} and {k2}")
    
    # Transpose a
    t = a.transpose(-1, -2)
    
    # Prepare output
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    num_blocks_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    t_allocated = torch.empty_like(t)
    
    transpose_matmul_kernel[(num_blocks_m, num_blocks_n)](
        a_ptr=a,
        b_ptr=b,
        t_ptr=t_allocated,
        out_ptr=out,
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return t_allocated, out

def replacement_func():
    return triton_transpose_matmul