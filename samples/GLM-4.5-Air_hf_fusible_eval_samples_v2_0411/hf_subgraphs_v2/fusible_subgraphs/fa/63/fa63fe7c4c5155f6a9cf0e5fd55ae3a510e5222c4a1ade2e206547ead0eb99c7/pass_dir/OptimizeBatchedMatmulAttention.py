import torch
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    batch_size,
    m,
    n,
    k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized batched matrix multiplication for attention computations.
    a: [batch_size, heads, m, k] 
    b: [batch_size, heads, k, n]
    c: [batch_size, heads, m, n]
    """
    # Matrix dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(3)
    
    # Compute program offsets within blocks
    num_pid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    group_id = pid_m // GROUP_SIZE_M
    group_size_m = min(num_pid_m, GROUP_SIZE_M)
    first_pid_m = group_id * GROUP_SIZE_M
    
    # offsets within the batch and head
    offset_b = pid_b * batch_size + (pid_h % batch_size) * m * n * k
    offset_m = first_pid_m * BLOCK_SIZE_M + (pid_m % group_size_m) * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    
    # Create pointers for matrices
    a_offset = offset_b + (offset_m * k + offset_n) * BLOCK_SIZE_N
    b_offset = offset_b + offset_m * k
    c_offset = offset_b + offset_m * n + offset_n
    
    # Create masks
    mask_m = offset_m + tl.arange(0, BLOCK_SIZE_M) < m
    mask_n = offset_n + tl.arange(0, BLOCK_SIZE_N) < n
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Accumulators
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k_offset in range(0, k, BLOCK_SIZE_K):
        # Load blocks
        a_block = tl.load(a_ptr + a_offset + k_offset, mask=mask, other=0.0)
        b_block = tl.load(b_ptr + b_offset + k_offset * BLOCK_SIZE_N, mask=mask, other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a_block, b_block, trans_b=True)
    
    # Store result
    tl.store(c_ptr + c_offset, accumulator.to(tl.float32), mask=mask)

@torch.fx.wrap
def optimized_batched_matmul(a, b):
    """
    Optimize batched matmul for attention patterns.
    a: [batch_size, heads, seq_len, head_dim]
    b: [batch_size, heads, head_dim, head_dim]  
    output: [batch_size, heads, seq_len, head_dim]
    """
    # Get tensor shapes
    batch_size, heads, m, k = a.shape
    _, _, _, n = b.shape
    assert k == b.shape[2], "Inner dimensions must match for matmul"
    
    # Output tensor
    c = torch.empty((batch_size, heads, m, n), dtype=a.dtype, device=a.device)
    
    # Block sizes tuned for attention patterns
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    grid_h = heads
    
    # Handle cases where dimensions are smaller than block sizes
    if grid_m == 0: grid_m = 1
    if grid_n == 0: grid_n = 1
    if grid_b == 0: grid_b = 1
    if grid_h == 0: grid_h = 1
    
    # Launch kernel
    batched_matmul_kernel[(grid_m // GROUP_SIZE_M, grid_n, grid_b, grid_h)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        batch_size=batch_size,
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c

def pattern(in_0, in_1, in_2):
    """Pattern matching for batched matmul in attention computation"""
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_2 = None
    tmp_4 = tmp_3.reshape(1, tmp_3.shape[1] * tmp_3.shape[2], tmp_3.shape[2], tmp_3.shape[2])
    tmp_3 = None
    tmp_5 = torch.functional.split(tmp_4, [tmp_4.shape[1]//4, tmp_4.shape[1]//2 - tmp_4.shape[1]//4, tmp_4.shape[1]//4], dim=1)
    tmp_4 = None
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_5 = None
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for matmul optimization"""
    return (in_0, in_1, in_2)

def replacement_func():
    """Return optimized batched matmul function"""
    return optimized_batched_matmul