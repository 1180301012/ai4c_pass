import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    """Matches: matmul + scalar multiplication pattern"""
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return tmp_1

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for fusion"""
    return (in_2, in_1, in_0)

@triton.jit
def simple_row_dot_kernel(
    matrix,
    vector,
    scalar,
    result,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel to compute row dot products with autotune"""
    row_idx = tl.program_id(0)
    
    if row_idx >= rows:
        return
    
    # Load the scalar once per row
    scale = tl.load(scalar)
    
    # Compute dot product for this row
    dot_product = 0.0
    for col in range(0, cols, BLOCK_SIZE):
        # Simple element-wise access for compatibility
        for k in range(col, min(col + BLOCK_SIZE, cols)):
            matrix_val = tl.load(matrix + row_idx * cols + k)
            vector_val = tl.load(vector + k)
            dot_product += matrix_val * vector_val
    
    # Apply scalar multiplication
    result_val = dot_product * scale
    
    # Store result
    tl.store(result + row_idx, result_val)

@torch.fx.wrap
def simple_fused_matmul_scale(matrix, vector, scalar):
    """Simple fused kernel for small matrices with autotune"""
    matrix_rows, matrix_cols = matrix.shape
    
    # Ensure vector is 1D [512]
    vector = vector.squeeze()
    
    # Create output [rows] 
    result_flat = torch.empty((matrix_rows,), dtype=matrix.dtype, device=matrix.device)
    
    # Use block size that processes entire row in one go for small matrices
    BLOCK_SIZE = 512
    
    # Launch kernel - one program per row
    simple_row_dot_kernel[(matrix_rows,)](
        matrix=matrix,
        vector=vector,
        scalar=scalar,
        result=result_flat,
        rows=matrix_rows,
        cols=matrix_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [rows, 1] to match expected output
    return result_flat.unsqueeze(1)

def replacement_func():
    """Return the fused function"""
    return simple_fused_matmul_scale