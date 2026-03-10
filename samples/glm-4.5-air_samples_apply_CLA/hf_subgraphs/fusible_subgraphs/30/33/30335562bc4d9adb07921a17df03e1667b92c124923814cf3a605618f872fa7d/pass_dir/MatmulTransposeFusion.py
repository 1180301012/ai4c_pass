import torch
import triton
import triton.language as tl

# Pattern matching: matmul followed by transpose(0, 2, 1)
def pattern(a, b):
    matmul_out = torch.matmul(a, b)
    transposed = matmul_out.permute(0, 2, 1)
    return transposed

# Argument extraction
def replacement_args(a, b):
    return (a, b)

# Optimized kernel that fuses matmul and transpose
@triton.jit
def matmul_transpose_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    m,
    k,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = (pid % num_pid_in_group) // num_pid_n
    group_size_m = tl.minimum(num_pid_m, 2048 // num_pid_n)
    group_m_start = group_id * group_size_m
    first_pid_m = first_pid_m + group_m_start
    pid_m = first_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    pid_n = tl.arange(0, BLOCK_SIZE_N)

    pid_m = pid_m % m
    pid_n = pid_n % n

    # Accumulator for the output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute range for reduction
    for k_idx in range(0, k, BLOCK_SIZE_K):
        # Load A with bounds checking
        a_ptrs = a_ptr + (pid_m[:, None] * k + tl.arange(0, BLOCK_SIZE_K)[None, :])
        a_mask = (pid_m[:, None] < m)[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B with bounds checking
        b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * n + pid_n[None, :])
        b_mask = (tl.arange(0, BLOCK_SIZE_K)[:, None] < k) & (pid_n[None, :] < n)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        accumulator = tl.dot(a, b, accumulator)
    
    # Store the result with bounds checking
    out_ptrs = out_ptr + (pid_n[:, None] * batch_size * m + pid_m[:, None] + tl.arange(0, batch_size)[None, None] * m)
    out_mask = (pid_n[:, None] < n)[:, None] & (pid_m[:, None] < m)[:, None] & (tl.arange(0, batch_size)[None, None] < batch_size)
    tl.store(out_ptrs, accumulator, mask=out_mask)

@torch.fx.wrap
def triton_matmul_transpose(a, b):
    batch_size, m, k = a.shape
    _, k2, n = b.shape
    
    assert k == k2, f"Inner dimension mismatch: {k} != {k2}"
    
    # For transpose fusion: (B, M, K) @ (B, K, N) -> (B, N, M)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    num_warps = 4
    grid = lambda META: (
        triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n, BLOCK_SIZE_N),
        batch_size,
        1,
    )
    
    # Allocate output tensor with transposed shape
    out = torch.empty((batch_size, n, m), dtype=a.dtype, device=a.device)
    
    # Launch the kernel
    matmul_transpose_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return triton_matmul_transpose