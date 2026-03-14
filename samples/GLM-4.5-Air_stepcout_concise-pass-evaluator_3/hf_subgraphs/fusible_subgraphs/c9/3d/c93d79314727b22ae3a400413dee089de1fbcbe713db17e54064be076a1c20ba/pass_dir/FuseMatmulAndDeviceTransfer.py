import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    # Match the matrix multiplication operation: tmp_2 = torch.matmul(in_2, in_3)
    # where in_2 is [2, 1152] and in_3 is [1152, 1], producing [2, 1] result
    tmp_2 = torch.matmul(in_2, in_3)
    return tmp_2

def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Optimized Triton kernel for small matrix-vector multiplication
@triton.jit
def matmul_vector_kernel(
    a_ptr,       # [M, K]
    b_ptr,       # [K] (flattened) 
    out_ptr,     # [M] (flattened)
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Each program handles one row of A
    m = pid
    
    # Check bounds
    if m >= M:
        return
    
    # Use multiple stages for better memory performance
    acc = 0.0
    for k in range(0, K, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < K
        
        # Load vectors with better memory access patterns
        a_vec = tl.load(a_ptr + m * K + offsets, mask=mask, other=0.0)
        b_vec = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(a_vec * b_vec)
    
    # Store the result
    tl.store(out_ptr + m, acc)

@torch.fx.wrap
def optimized_matmul_vector(a, b):
    # Input shapes: a [M, K] = [2, 1152], b [K, 1] = [1152, 1] -> output [M, 1] = [2, 1]
    M, K = a.shape
    
    # Use optimal block size for this specific case
    BLOCK_SIZE = 256  # Smaller block size for smaller matrices
    
    # Create output tensor - we'll create it as [M, 1] and flatten for kernel
    out_flat = torch.empty((M,), dtype=a.dtype, device=a.device)
    
    # Launch kernel - one program per row of A
    num_programs = M
    matmul_vector_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b.flatten(),  # Flatten [K, 1] to [K]
        out_ptr=out_flat,   # Store flattened result
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [M, 1]
    return out_flat.reshape(M, 1)

def replacement_func():
    return optimized_matmul_vector