import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, bias=None):
    """Match the linear operation from torch.nn.functional.linear"""
    # This matches torch.nn.functional.linear(in_0, in_1, None)
    result = torch.nn.functional.linear(in_0, in_1, bias)
    return result

def replacement_args(in_0, in_1, bias=None):
    return (in_0, in_1, bias)

@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Matrix multiplication kernel for linear transformation"""
    # Program ID for row block
    pid_m = tl.program_id(0)
    # Program ID for column block
    pid_n = tl.program_id(1)
    
    # Compute row and column offsets
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets for iteration within the block
    offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary conditions
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Allocate accumulator in shared memory
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offset for K dimension
        k_start = k
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Create K mask
        mask_k = k_start < k_end
        
        # Load X data (input matrix)
        x_ptrs = x_ptr + offsets_m[:, None] * K + k_start
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k, other=0.0)
        
        # Load weight data
        weight_ptrs = weight_ptr + k_start[:, None] * N + offsets_n[None, :]
        weight = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiplication
        accumulator += x[:, :, None] * weight[None, None, :]
    
    # Sum accumulator and handle bias if present
    accumulator = tl.sum(accumulator, axis=1)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)
        accumulator += bias[None, :]
    
    # Store result
    out_ptrs = out_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(out_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_linear(x, weight, bias=None):
    """Optimized linear operation using Triton"""
    M, K = x.shape
    N = weight.shape[0]
    
    # Set optimal block sizes based on matrix dimensions
    if M <= 128 and N <= 128:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
    elif M > 512 or N > 512:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
    
    # Compute number of blocks
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output tensor
    out = torch.empty((M, N), dtype=tl.float32, device=x.device)
    
    # Launch kernel
    linear_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return optimized_linear