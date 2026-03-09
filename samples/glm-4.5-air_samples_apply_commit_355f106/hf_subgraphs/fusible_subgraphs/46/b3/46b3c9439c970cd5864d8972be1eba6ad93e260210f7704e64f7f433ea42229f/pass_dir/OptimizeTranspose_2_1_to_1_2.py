import torch
import triton
import triton.language as tl

def pattern(x):
    # Match transpose pattern from [2, 1] to [1, 2]
    return x.t()

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the transposed matrix
    pid = tl.program_id(0)
    
    # Calculate offsets for transposed matrix
    # Original: [M, N], Transposed: [N, M]
    m_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_offset = tl.arange(0, M)
    
    # Create mask to ensure we don't go out of bounds
    mask = m_offset < N
    
    # Load input data in transposed order
    input_ptrs = input_ptr + (n_offset[:, None] * N + m_offset[None, :])
    input_data = tl.load(input_ptrs, mask=mask[None, :], other=0.0)
    
    # Store output in correct order
    output_ptrs = output_ptr + (m_offset[None, :] * M + n_offset[:, None])
    tl.store(output_ptrs, input_data, mask=mask[None, :])

@torch.fx.wrap
def optimized_transpose_func(x):
    # Input matrix dimensions: M=2 rows, N=1 column
    M, N = x.shape
    transposed_M, transposed_N = N, M  # Transposed dimensions: [1, 2]
    
    # Optimal block size for small transposes
    if N <= 32:
        BLOCK_SIZE = 32
    else:
        BLOCK_SIZE = 64
    
    # Calculate grid size (only need to iterate over the smaller dimension N)
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with transposed shape
    out = torch.empty((transposed_M, transposed_N), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    optimized_transpose_kernel[(grid_size,)](
        input_ptr=x,
        output_ptr=out,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_transpose_func