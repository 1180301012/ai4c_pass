import torch
import triton
import triton.language as tl

def pattern(x, dim1, dim2, dim3):
    """Pattern to match view + softmax + dropout operations"""
    tmp_3 = x.view(dim1, dim2, dim3)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5

def replacement_args(x, dim1, dim2, dim3):
    return (x, dim1, dim2, dim3)

@triton.jit
def softmax_dropout_kernel(
    output_ptr,
    input_ptr,
    n_rows,
    n_cols,
    dropout_prob: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized softmax + dropout kernel with better memory access patterns"""
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    
    # Calculate row range for this program
    row_start = row_id * BLOCK_SIZE_M
    row_end = tl.minimum(row_start + BLOCK_SIZE_M, n_rows)
    
    # Initialize accumulator for max and sum
    row_max = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + -tl.inf
    row_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Process input in chunks for better memory efficiency
    offset_n = col_id * BLOCK_SIZE_N
    
    # Load max and sum reduction for softmax
    for m_offset in range(0, row_end - row_start, BLOCK_SIZE_N):
        current_offset = offset_n + m_offset
        
        # Check bounds and load data
        mask = (row_start + tl.arange(0, BLOCK_SIZE_M) < row_end) & \
               (current_offset + tl.arange(0, BLOCK_SIZE_N) < n_cols)
        
        # Load data with proper masking
        data = tl.load(
            input_ptr + (row_start - row_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64) * n_cols + \
            current_offset + tl.arange(0, BLOCK_SIZE_N),
            mask=mask,
            other=-tl.inf
        )
        
        # Update max for softmax stability
        row_max = tl.maximum(row_max, tl.max(data, 1))
        row_sum = row_sum + tl.sum(tl.exp(data - row_max[:, None]), 1)
    
    # Compute final softmax and apply dropout
    for m_offset in range(0, row_end - row_start, BLOCK_SIZE_N):
        current_offset = offset_n + m_offset
        
        mask = (row_start + tl.arange(0, BLOCK_SIZE_M) < row_end) & \
               (current_offset + tl.arange(0, BLOCK_SIZE_N) < n_cols)
        
        data = tl.load(
            input_ptr + (row_start - row_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64) * n_cols + \
            current_offset + tl.arange(0, BLOCK_SIZE_N),
            mask=mask,
            other=-tl.inf
        )
        
        # Apply softmax
        softmax_vals = tl.exp(data - row_max[:, None]) / row_sum[:, None]
        
        # Apply dropout during training (p=0.1 for dropout, so keep_rate=0.9)
        keep_rate = 1.0 - dropout_prob
        rand_vals = tl.rand(tl.shape(softmax_vals))
        dropout_mask = rand_vals < keep_rate
        result = softmax_vals * dropout_mask
        
        # Store result
        tl.store(
            output_ptr + (row_start - row_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64) * n_cols + \
            current_offset + tl.arange(0, BLOCK_SIZE_N),
            result,
            mask=mask
        )

@torch.fx.wrap
def optimized_softmax_with_dropout(x, dim1, dim2, dim3):
    """Wrapper for optimized softmax + dropout with view transformation"""
    # Reshape input tensor
    reshaped_x = x.view(dim1, dim2, dim3)
    
    # Get dimensions for 3D tensor (assuming last two dimensions are for softmax)
    n_rows = dim1 * dim2  # Treat as flattened for softmax computation
    n_cols = dim3
    
    # Create output tensor
    out = torch.empty_like(reshaped_x)
    
    # Optimize block sizes based on tensor dimensions
    BLOCK_SIZE_M = 32  # Block size for rows
    BLOCK_SIZE_N = 128  # Block size for columns
    
    # Calculate grid sizes
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with optimized grid
    softmax_dropout_kernel[grid_m, grid_n](
        output_ptr=out,
        input_ptr=reshaped_x,
        n_rows=n_rows,
        n_cols=n_cols,
        dropout_prob=0.1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_softmax_with_dropout