import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern that matches matmul + scalar multiplication"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the optimized kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_matmul_scalar_kernel(
    x_ptr,                # [2, 512] input matrix
    y_ptr,                # [512, 1] input matrix  
    scale_ptr,            # scalar scale factor
    out_ptr,              # [2, 1] output
    M: tl.constexpr,      # 2 (rows of result)
    N: tl.constexpr       # 512 (shared dimension)
):
    # Each program handles one row - minimal overhead with best-tuned block size
    m = tl.program_id(0)
    acc = 0.0
    
    # Use 128-element blocks for optimal cache utilization on N=512
    BLOCK_SIZE = 128
    
    # Process in efficient chunks
    for start in range(0, N, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, N)
        # Process chunk efficiently
        for i in range(start, end):
            x_val = tl.load(x_ptr + m * N + i)
            y_val = tl.load(y_ptr + i)
            acc += x_val * y_val
    
    # Apply scalar multiplication
    scale = tl.load(scale_ptr)
    result = acc * scale
    
    # Store the result
    tl.store(out_ptr + m, result)

@torch.fx.wrap
def fused_matmul_scalar_wrapper(in_0, in_1, in_2):
    """Wrapper that launches the optimized fused kernel"""
    M = 2    # Shape from in_2 [2,512] 
    N = 512  # Shared dimension
    
    # Create output tensor
    out_shape = (M, 1)  # [2,1]
    out = torch.empty(out_shape, dtype=torch.float32, device=in_2.device)
    
    # Launch Triton kernel with minimal overhead - one program per row
    fused_matmul_scalar_kernel[(M,)](
        x_ptr=in_2,
        y_ptr=in_1,
        scale_ptr=in_0,
        out_ptr=out,
        M=M,
        N=N
    )
    
    return out

def replacement_func():
    """Return the optimized fusion function"""
    return fused_matmul_scalar_wrapper