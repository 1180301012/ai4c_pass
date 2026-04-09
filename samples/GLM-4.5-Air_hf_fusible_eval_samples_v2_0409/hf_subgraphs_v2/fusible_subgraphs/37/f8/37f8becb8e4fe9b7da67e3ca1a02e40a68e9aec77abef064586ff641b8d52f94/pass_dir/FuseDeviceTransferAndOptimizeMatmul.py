import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Simple pattern for matrix multiplication optimization"""
    result = torch.matmul(x, y)
    return result

def replacement_args(in_2, in_3):
    """Extract arguments for the replacement function"""
    return (in_2, in_3)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Simplified matrix multiplication kernel for small matrices"""
    # Program ID
    pid = tl.program_id(0)
    
    # Determine row and column for this program
    row = pid // N
    col = pid % N
    
    # Check if we're within bounds
    if row >= M or col >= N:
        return
    
    # Compute the result for this output element
    result = 0.0
    
    # Vectorized computation for the matrix multiplication
    for k in range(0, K, BLOCK_SIZE):
        # Load elements from matrix A (current row)
        a_offs = row * K + k + tl.arange(0, BLOCK_SIZE)
        a_mask = k + tl.arange(0, BLOCK_SIZE) < K
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        
        # Load elements from matrix B (current column)
        b_offs = (k + tl.arange(0, BLOCK_SIZE)) * N + col
        b_mask = k + tl.arange(0, BLOCK_SIZE) < K
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
        
        # Compute dot product for this chunk
        result += tl.sum(a * b)
    
    # Store the result
    tl.store(out_ptr + row * N + col, result, mask=(row < M) & (col < N))

@torch.fx.wrap
def optimized_matmul(a, b):
    """High-performance matrix multiplication with device transfer fusion"""
    M, K = a.shape
    _, N = b.shape
    
    # Get dtype from input tensors
    dtype = a.dtype
    if dtype == torch.float16:
        output_dtype = torch.float16
    elif dtype == torch.bfloat16:
        output_dtype = torch.bfloat16
    else:
        output_dtype = torch.float32
    
    # Create output tensor on GPU
    out = torch.empty((M, N), dtype=output_dtype, device='cuda')
    
    # Block size optimized for small matrices like [2, 768] x [768, 1] or [2, 1152] x [1152, 1]
    BLOCK_SIZE = 64
    
    # Number of programs (one program per output element)
    num_programs = M * N
    
    # Launch kernel
    matmul_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        M=M, N=N, K=K,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def fuse_device_transfers(in_0, in_1):
    """Fuse device transfers for small scalar tensors"""
    # Transfer both tensors to GPU in one operation
    out_0 = in_0.to(device(type='cuda'))
    out_1 = in_1.to(device(type='cuda'))
    
    return out_0, out_1

def replacement_func():
    """Return optimized matrix multiplication function"""
    return optimized_matmul