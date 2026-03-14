import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Pattern matching for the entire computation: matmul + scalar + transpose"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    tmp_0 = None
    tmp_2 = tmp_1.T
    return tmp_1, tmp_2

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for replacement"""
    return (in_2, in_1, in_0)

@triton.jit
def full_optimized_kernel(
    x_ptr,      # in_2: [2, 512]
    y_ptr,      # in_1: [512, 1] 
    scalar_val, # in_0: scalar
    out1_ptr,   # tmp_1: [2, 1]
    out2_ptr,   # tmp_2: [1, 2]
    M: tl.constexpr,  # 2 (rows)
    K: tl.constexpr,  # 512 (inner dimension)
    N: tl.constexpr,  # 1 (columns)
):
    """Kernel that computes matmul + scalar + optimized transpose in one pass"""
    
    # Each program handles one output element of the matmul result
    i = tl.program_id(0)
    
    # Bounds check
    if i >= M:
        return
    
    # Load scalar
    scalar = tl.load(scalar_val)
    
    # Compute dot product for this row
    dot_product = 0.0
    for k in range(K):
        k_mask = (k < K)
        x = tl.load(x_ptr + i * K + k, mask=k_mask, other=0.0)
        y = tl.load(y_ptr + k * N, mask=k_mask, other=0.0)
        dot_product += x * y
    
    # Store matmul result * scalar
    result = dot_product * scalar
    tl.store(out1_ptr + i * N, result)
    
    # Store transposed result directly (row i becomes column i)
    if N == 1:  # Special case for [M, 1] -> [1, M] transpose
        tl.store(out2_ptr + i, result)

@torch.fx.wrap
def full_optimized_computation(in_2, in_1, in_0):
    """Optimized computation: matmul + scalar + transpose"""
    M, K = in_2.shape
    N = in_1.shape[1]
    
    # Create output tensors
    out1 = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)  # tmp_1
    out2 = torch.empty((N, M), dtype=in_2.dtype, device=in_2.device)  # tmp_2 (transpose)
    
    # Launch kernel
    grid = (M,)
    full_optimized_kernel[grid](
        in_2,
        in_1,
        in_0,
        out1,
        out2,
        M, K, N
    )
    
    return out1, out2

def replacement_func():
    """Return the full optimized function"""
    return full_optimized_computation