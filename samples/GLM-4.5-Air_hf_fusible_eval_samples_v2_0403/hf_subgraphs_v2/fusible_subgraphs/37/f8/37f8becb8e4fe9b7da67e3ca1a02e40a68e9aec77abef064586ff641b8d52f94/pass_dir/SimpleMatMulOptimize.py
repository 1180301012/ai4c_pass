import torch
import triton
import triton.language as tl

# Pattern matching function - matches just the matmul operation
def pattern(in_2, in_3):
    """Match just the matmul operation"""
    matmul = torch.matmul(in_2, in_3)
    return matmul

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Very simple Triton kernel for matrix multiplication
@triton.jit
def simple_matmul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    K,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    
    # Determine row and column based on program ID
    num_rows = tl.cdiv(M, BLOCK_SIZE)
    row = pid // num_rows
    col = pid % num_rows
    
    # Check bounds
    if row * BLOCK_SIZE >= M or col * BLOCK_SIZE >= N:
        return
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Matrix multiplication computation
    for k in range(0, K, BLOCK_SIZE):
        # Load tiles from x and y
        x_tile = tl.load(x_ptr + (row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] * K + 
                        tl.arange(0, BLOCK_SIZE)[None, :] + k,
                        mask=(row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < M and 
                             (tl.arange(0, BLOCK_SIZE)[None, :] + k) < K,
                        other=0.0)
        
        y_tile = tl.load(y_ptr + (col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] * N + 
                        tl.arange(0, BLOCK_SIZE)[None, :] + k,
                        mask=(col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < N and 
                             (tl.arange(0, BLOCK_SIZE)[None, :] + k) < K,
                        other=0.0)
        
        # Compute dot product
        accumulator += tl.dot(x_tile, y_tile)
    
    # Store result
    tl.store(out_ptr + (row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] * N + 
             col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :],
             accumulator,
             mask=(row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))[:, None] < M and 
                  (col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < N)

# Simple replacement function that avoids all device API calls
@torch.fx.wrap
def simple_replacement(in_2, in_3):
    # Perform the optimized matrix multiplication
    # Note: We assume the tensors are already on the correct device
    # as indicated by the weight_meta data
    
    M, K = in_2.shape
    _, N = in_3.shape
    
    # Create output tensor with same properties as one of the inputs
    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    
    # Use simple block size for now (can be optimized later)
    BLOCK_SIZE = 32
    
    # Calculate grid size using regular Python math
    rows_per_block = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    cols_per_block = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_size = rows_per_block * cols_per_block
    
    # Launch kernel only if grid size is reasonable
    if grid_size > 0:
        simple_matmul_kernel[(grid_size,)](
            in_2, in_3, out, M, K, N, BLOCK_SIZE
        )
    
    # Return only the matmul result since that's what our pattern matches
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_replacement