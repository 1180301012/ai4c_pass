import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    tmp_2 = torch.matmul(in_2, in_3)
    return tmp_2

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def small_matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized small matrix multiplication kernel for small matrices like 2x1152 @ 1152x1"""
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= M or pid_n >= N:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over K dimension with fixed block size
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = tl.arange(0, BLOCK_SIZE_K) < (K - k)
        
        # Load x_block [BLOCK_SIZE_K] = x[pid_m, k:k+BLOCK_SIZE_K]
        x_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        x_vals = tl.load(x_ptr + pid_m * K + x_offsets, mask=k_mask, other=0.0)
        
        # Load y_block [BLOCK_SIZE_K] = y[k:k+BLOCK_SIZE_K, pid_n]
        y_vals = tl.load(y_ptr + (k + tl.arange(0, BLOCK_SIZE_K)) * N + pid_n, mask=k_mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(x_vals * y_vals)
    
    # Store result
    tl.store(out_ptr + pid_m * N + pid_n, acc)

@torch.fx.wrap
def optimized_matmul(x, y):
    M, K = x.shape
    N = y.shape[1]
    
    # Optimize for small matrices like 2x1152 @ 1152x1
    # Use one thread per output element for minimal overhead
    grid_m = M  # One thread per row
    grid_n = N  # One thread per column
    
    # Allocate output tensor
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    
    # Choose block size for K dimension - use 64 which divides evenly into 1152
    BLOCK_SIZE_K = 64
    
    # Use optimized kernel designed for small matrices
    small_matmul_kernel[(grid_m, grid_n)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return optimized_matmul