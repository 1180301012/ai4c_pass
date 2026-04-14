import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Pattern: matrix multiplication followed by transpose of last two dimensions"""
    matmul_result = torch.matmul(a, b)
    transposed_result = matmul_result.permute(0, 2, 1)
    return transposed_result

def replacement_args(a, b):
    """Extract arguments for the optimized matmul+transpose operation"""
    return (a, b)

@triton.jit
def optimized_matmul_transpose_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    a_rows,
    a_cols,
    b_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized kernel that computes matmul and transpose in one pass"""
    # Each program handles one output tile
    batch_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)
    
    # Compute offsets within batch
    batch_a_offset = batch_id * a_rows * a_cols
    batch_b_offset = batch_id * a_cols * b_cols
    batch_out_offset = batch_id * b_cols * a_rows
    
    # Compute starting positions
    a_offset = batch_a_offset + m_block * BLOCK_SIZE_M * a_cols
    b_offset = batch_b_offset + n_block * BLOCK_SIZE_N
    out_offset = batch_out_offset + n_block * BLOCK_SIZE_N * a_rows + m_block * BLOCK_SIZE_M
    
    # Initialize accumulator for this output tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension (matmul dimension)
    for k in range(0, a_cols, BLOCK_SIZE_K):
        # Load a tile from matrix A (transposed for better memory access)
        a_offset_k = a_offset + k
        a_tile = tl.load(
            a_ptr + a_offset_k + (tl.arange(0, BLOCK_SIZE_M)[:, None] * a_cols + tl.arange(0, BLOCK_SIZE_K)[None, :]),
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (a_rows - m_block * BLOCK_SIZE_M)) & 
                 (tl.arange(0, BLOCK_SIZE_K)[None, :] < (a_cols - k)),
            other=0.0
        )
        
        # Load a tile from matrix B
        b_offset_k = b_offset + k * b_cols
        b_tile = tl.load(
            b_ptr + b_offset_k + (tl.arange(0, BLOCK_SIZE_K)[:, None] * b_cols + tl.arange(0, BLOCK_SIZE_N)[None, :]),
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < (a_cols - k)) & 
                 (tl.arange(0, BLOCK_SIZE_N)[None, :] < (b_cols - n_block * BLOCK_SIZE_N)),
            other=0.0
        )
        
        # Matrix multiply: accumulator += a_tile @ b_tile^T
        # Note: We're computing (a @ b)^T = b^T @ a^T, but we transpose a to get optimal memory access
        accumulator += tl.dot(a_tile, b_tile, trans_b=True)
    
    # Store result directly in transposed position [batch, b_cols, a_rows]
    out_ptr_tile = out_ptr + out_offset
    tl.store(
        out_ptr_tile + (tl.arange(0, BLOCK_SIZE_M)[:, None] * b_cols + tl.arange(0, BLOCK_SIZE_N)[None, :]),
        accumulator,
        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (a_rows - m_block * BLOCK_SIZE_M)) & 
             (tl.arange(0, BLOCK_SIZE_N)[None, :] < (b_cols - n_block * BLOCK_SIZE_N))
    )

@torch.fx.wrap
def optimized_matmul_transpose(a, b):
    """Wrapper function for optimized matmul+transpose kernel"""
    # Get tensor shapes
    batch_size = a.shape[0]
    a_rows = a.shape[1]
    a_cols = a.shape[2]
    b_cols = b.shape[2]
    
    # Optimal block sizes for GPU
    BLOCK_SIZE_M = 64   # Output tile size M (corresponds to original a_rows)
    BLOCK_SIZE_N = 64   # Output tile size N (corresponds to original b_cols) 
    BLOCK_SIZE_K = 32   # Reduction dimension (corresponds to original a_cols)
    
    # Calculate grid dimensions
    m_grid = (a_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_grid = (b_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor with transposed dimensions [batch, b_cols, a_rows]
    output = torch.empty((batch_size, b_cols, a_rows), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (
        batch_size,
        m_grid,
        n_grid,
    )
    
    optimized_matmul_transpose_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=output,
        batch_size=batch_size,
        a_rows=a_rows,
        a_cols=a_cols,
        b_cols=b_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    """Return the optimized matmul+transpose function"""
    return optimized_matmul_transpose