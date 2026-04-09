import torch
import triton
import triton.language as tl

@triton.jit
def simple_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple matrix multiplication - treat each matrix as 1D and do element-wise ops
    idx = tl.program_id(0)
    
    # Handle each element in the output matrix
    i = idx // N
    j = idx % N
    
    # Bounds check
    if i >= M or j >= N:
        return
    
    # Initialize accumulator for this output element
    acc = 0.0
    
    # Dot product over K dimension
    for k in range(K):
        # Load elements from A and B (always in bounds in this loop)
        a_val = tl.load(a_ptr + i * K + k)
        b_val = tl.load(b_ptr + k * N + j)
        
        # Multiply and accumulate
        acc += a_val * b_val
    
    # Store result
    tl.store(out_ptr + i * N + j, acc.to(b_ptr.dtype.element_ty))

@torch.fx.wrap
def simple_matmul(a, b):
    M, N = a.shape[0], a.shape[1]
    K = b.shape[1]
    
    BLOCK_SIZE = 256  # Process 256 elements per program
    
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Total number of elements in output matrix
    total_elements = M * N
    grid_size = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    simple_matmul_kernel[grid_size](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x, y):
    return torch.matmul(x, y)

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return simple_matmul