import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Pattern matching: matmul followed by scalar multiplication"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for replacement"""
    return (in_2, in_1, in_0)

@triton.jit
def fused_matmul_scalar_kernel(
    x_ptr,      # in_2: [2, 512]
    y_ptr,      # in_1: [512, 1] 
    scalar_val, # in_0: scalar
    out_ptr,    # output: [2, 1]
    M: tl.constexpr,  # 2 (rows)
    K: tl.constexpr,  # 512 (inner dimension)
    N: tl.constexpr,  # 1 (columns)
):
    # Each program handles one output element [i, j]
    i = tl.program_id(0)
    
    # Check bounds
    if i >= M:
        return
    
    # Load scalar once (no bounds needed for scalar)
    scalar = tl.load(scalar_val)
    
    # Initialize accumulator for this element
    accumulator = 0.0
    
    # Vectorized dot product with scalar fused
    for k in range(0, K):
        # Create masks for bounds checking
        k_mask = k < K
        
        # Load x element [i, k]
        x = tl.load(x_ptr + i * K + k, mask=k_mask, other=0.0)
        
        # Load y element [k, 0] (since N=1, j is always 0)
        y = tl.load(y_ptr + k * N, mask=k_mask, other=0.0)
        
        # Accumulate with scalar multiplication
        accumulator += x * y * scalar
    
    # Store result
    tl.store(out_ptr + i * N, accumulator)

@torch.fx.wrap
def fused_matmul_scalar(in_2, in_1, in_0):
    """Fused kernel combining matmul and scalar multiplication"""
    M, K = in_2.shape
    N = in_1.shape[1]
    
    # Calculate grid size (one program per output element)
    grid = (M,)
    
    # Create output tensor
    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    fused_matmul_scalar_kernel[grid](
        in_2,
        in_1,
        in_0,
        out,
        M, K, N
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_matmul_scalar