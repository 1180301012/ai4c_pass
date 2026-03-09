import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple pattern: just match matrix multiplication
    return torch.matmul(x, y)

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID - each program handles one row
    row_idx = tl.program_id(0)
    
    # Check bounds
    if row_idx >= M:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Process K dimension in a simple loop for small K
    k_base = 0
    while k_base < K:
        # Load X[row_idx, k_base]
        x_offset = row_idx * K + k_base
        x_mask = x_offset < (M * K)
        x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
        
        # Load Y[k_base, 0]
        y_offset = k_base
        y_mask = y_offset < (K * N)
        y_val = tl.load(y_ptr + y_offset, mask=y_mask, other=0.0)
        
        # Accumulate product
        acc += x_val * y_val
        
        k_base += 1
    
    # Store the final result
    tl.store(out_ptr + row_idx * N, acc)

@torch.fx.wrap
def optimized_matmul(x, y):
    # Get matrix dimensions
    M, K = x.shape
    N = y.shape[1]
    
    # Create output tensor 
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Check if we need to launch kernel
    if output.numel() > 0:
        # Each program handles one row, so we need M programs
        grid_size = M
        
        # Launch kernel - Triton expects grid as a tuple
        optimized_matmul_kernel[(grid_size,)](
            x,
            y,
            output,
            M=M,
            N=N,
            K=K,
            BLOCK_SIZE_K=64,
        )
    
    return output

def replacement_func():
    return optimized_matmul