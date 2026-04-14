import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Pattern to match matmul followed by scalar multiplication"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for the fused matmul + scalar multiplication kernel"""
    return (in_2, in_1, in_0)

@triton.jit
def fused_matmul_scalar_kernel(
    x_ptr,      # in_2: [M, K]
    y_ptr,      # in_1: [K, N] 
    scalar_ptr,
    out_ptr,    # tmp_1: [M, N]
    M,          # 2
    K,          # 512
    N,          # 1
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one row of the output
    m = tl.program_id(0)
    
    # Load scalar
    scalar = tl.load(scalar_ptr)
    
    # Load the current row of x [K]
    x_offsets = m * K + tl.arange(0, BLOCK_SIZE_K)
    x = tl.load(x_ptr + x_offsets, mask=tl.arange(0, BLOCK_SIZE_K) < K, other=0.0)
    
    # Load the entire column of y [K] (since N=1)
    y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE_K), mask=tl.arange(0, BLOCK_SIZE_K) < K, other=0.0)
    
    # Compute dot product (reduction over K)
    acc = tl.sum(x * y) * scalar
    
    # Store result in [M, N] layout at position (m, 0)
    tl.store(out_ptr + m * N + 0, acc)

@torch.fx.wrap
def fused_matmul_scalar_func(in_2, in_1, in_0):
    """Wrapper function to launch the fused kernel"""
    M, K = in_2.shape
    N = in_1.shape[1]
    
    # Output buffer
    tmp_1 = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel - one program per row since N=1
    grid = (M,)
    BLOCK_SIZE_K = 128  # Process 128 elements at a time for good performance
    
    fused_matmul_scalar_kernel[grid](
        in_2,
        in_1, 
        in_0,
        tmp_1,
        M, K, N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return tmp_1

def replacement_func():
    """Return the fused kernel function"""
    return fused_matmul_scalar_func