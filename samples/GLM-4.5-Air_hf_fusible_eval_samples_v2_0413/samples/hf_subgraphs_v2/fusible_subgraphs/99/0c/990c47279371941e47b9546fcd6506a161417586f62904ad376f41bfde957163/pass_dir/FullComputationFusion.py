import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern: Full computation chain from input to final output"""
    # Full computation: scale -> softmax -> matmul -> transpose
    tmp_0 = 0.0625 * x
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, y)
    tmp_3 = matmul.permute(0, 2, 1)
    return (tmp_3,)

def replacement_args(x, y):
    """Extract arguments for the full fused computation"""
    return (x, y)

@triton.jit
def full_fusion_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    x_rows,
    x_cols,
    y_cols,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fully fused kernel that scales, applies softmax, computes matmul and transposes"""
    # Each program handles one batch and one output tile
    batch_id = tl.program_id(0)
    m_block = tl.program_id(1)  # Output rows (corresponds to y_cols)
    n_block = tl.program_id(2)  # Output cols (corresponds to x_rows)
    
    # Compute batch offsets
    x_batch_offset = batch_id * x_rows * x_cols
    y_batch_offset = batch_id * x_cols * y_cols
    out_batch_offset = batch_id * y_cols * x_rows
    
    # Starting positions
    x_offset = x_batch_offset + n_block * BLOCK_SIZE_X * x_cols  # For columns of input x
    y_offset = y_batch_offset + m_block * BLOCK_SIZE_Y  # For rows of input y
    out_offset = out_batch_offset + m_block * BLOCK_SIZE_Y * x_rows + n_block * BLOCK_SIZE_X
    
    # accumulator for softmax and matmul
    exp_sum = tl.zeros((BLOCK_SIZE_Y,), dtype=tl.float32)
    output_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Process one row of softmax at a time
    for x_row_idx in range(0, min(BLOCK_SIZE_X, x_rows - n_block * BLOCK_SIZE_X)):
        # Load x row for softmax
        x_row_ptr = x_ptr + x_batch_offset + (n_block * BLOCK_SIZE_X + x_row_idx) * x_cols
        x_vals = tl.load(
            x_row_ptr + tl.arange(0, BLOCK_SIZE_Y),
            mask=tl.arange(0, BLOCK_SIZE_Y) < min(BLOCK_SIZE_Y, x_cols),
            other=float('-inf')
        )
        
        # Apply scale and softmax
        scaled_x = x_vals * 0.0625
        max_val = tl.max(scaled_x)
        exp_x = tl.exp(scaled_x - max_val)
        exp_x_sum = tl.sum(exp_x)
        softmax_vals = exp_x / exp_x_sum
        
        # Store softmax results and accumulate matmul sums
        tl.store(
            x_row_ptr + tl.arange(0, BLOCK_SIZE_Y),
            softmax_vals,
            mask=tl.arange(0, BLOCK_SIZE_Y) < min(BLOCK_SIZE_Y, x_cols)
        )
        
        # Accumulate matmul results
        for y_col_idx in range(0, BLOCK_SIZE_Y):
            if y_col_idx < y_cols - m_block * BLOCK_SIZE_Y:
                # Load y column
                y_col_ptr = y_ptr + y_batch_offset + y_col_idx * y_cols + m_block * BLOCK_SIZE_Y
                y_vals = tl.load(
                    y_col_ptr + tl.arange(0, BLOCK_SIZE_X),
                    mask=(tl.arange(0, BLOCK_SIZE_X) + n_block * BLOCK_SIZE_X) < x_rows,
                    other=0.0
                )
                
                # Compute dot product with softmax row
                dot_product = tl.dot(softmax_vals[:y_cols], y_vals)
                output_vals[y_col_idx, x_row_idx] = dot_product
    
    # Store final result directly in transposed position
    out_ptr_tile = out_ptr + out_offset
    tl.store(
        out_ptr_tile + (tl.arange(0, BLOCK_SIZE_Y)[:, None] * x_rows + tl.arange(0, BLOCK_SIZE_X)[None, :]),
        output_vals,
        mask=(tl.arange(0, BLOCK_SIZE_Y)[:, None] < min(BLOCK_SIZE_Y, y_cols - m_block * BLOCK_SIZE_Y)) & 
             (tl.arange(0, BLOCK_SIZE_X)[None, :] < min(BLOCK_SIZE_X, x_rows - n_block * BLOCK_SIZE_X))
    )

@torch.fx.wrap
def full_computation_fusion(x, y):
    """Wrapper function for full computation fusion kernel"""
    # Get tensor shapes
    batch_size = x.shape[0]
    x_rows = x.shape[1]      # 8192
    x_cols = x.shape[2]      # 19  
    y_cols = y.shape[2]      # 256
    
    # Optimal block sizes for GPU architecture
    BLOCK_SIZE_X = 256  # Block size for x dimension (rows)
    BLOCK_SIZE_Y = 64   # Block size for y dimension (columns)
    BLOCK_SIZE_K = 32   # Block size for reduction dimension
    
    # Calculate grid dimensions
    x_grid = (x_rows + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    y_grid = (y_cols + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor with transposed dimensions [batch, y_cols, x_rows]
    output = torch.empty((batch_size, y_cols, x_rows), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (
        batch_size,
        y_grid,
        x_grid,
    )
    
    full_fusion_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        batch_size=batch_size,
        x_rows=x_rows,
        x_cols=x_cols,
        y_cols=y_cols,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    """Return the full computation fusion function"""
    return full_computation_fusion