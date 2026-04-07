import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(x):
    """
    Matches: softmax followed by dropout (with training=False)
    This is common in transformer inference where dropout is just scaling by (1-p)
    """
    # Softmax operation on last dimension
    tmp_4 = torch.nn.functional.softmax(x, dim=-1)
    # Dropout with training=False is just scaling by (1-p) = 0.9
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5

# Fusion of softmax and dropout operations
@triton.jit
def softmax_dropout_kernel(
    x_ptr,           # Pointer to input tensor
    out_ptr,         # Pointer to output tensor  
    rows,            # Number of rows (first dimension)
    cols,            # Number of columns (last dimension)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel fusing softmax and dropout scaling
    Softmax is computed row-wise (across the last dimension)
    """
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate row start and column offsets
    row_start = pid_m * BLOCK_SIZE_M
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Row mask
    row_mask = row_start + tl.arange(0, BLOCK_SIZE_M) < rows
    
    # Load input data for this block
    x = tl.load(x_ptr + (row_start[:, None] * cols + col_offsets[None, :]), 
                mask=row_mask[:, None] and col_offsets[None, :] < cols, 
                other=-float('inf'))
    
    # Compute max for numerical stability (softmax step 1)
    max_vals = tl.maximum(x, -float('inf'))
    
    # Compute exponential (softmax step 2)
    exp_vals = tl.exp(x - max_vals)
    
    # Compute sum for normalization (softmax step 3)
    sum_vals = tl.sum(exp_vals, axis=1)
    
    # Compute softmax (softmax step 4)
    softmax_vals = exp_vals / sum_vals[:, None]
    
    # Apply dropout scaling: multiply by (1-p) = 0.9
    dropout_scale = 0.9  # 1 - 0.1 dropout probability
    output_vals = softmax_vals * dropout_scale
    
    # Store result
    tl.store(out_ptr + (row_start[:, None] * cols + col_offsets[None, :]), 
             output_vals, 
             mask=row_mask[:, None] and col_offsets[None, :] < cols)

@torch.fx.wrap
def fused_softmax_dropout(x):
    """
    Fusion of softmax and dropout operations into a single kernel
    This reduces memory bandwidth usage and improves GPU utilization
    """
    # Handle different tensor shapes: we assume softmax on last dimension
    if x.dim() == 4:  # Shape like [1, C, H, W] from transformer models
        # Reshape to [C, H*W] for efficient softmax computation
        original_shape = x.shape
        x_reshaped = x.reshape(original_shape[1], -1)  # [C, H*W]
        rows, cols = x_reshaped.shape
    else:  # Shape like [C, H, W] after view operation
        # Reshape to [C, H*W] for efficient softmax computation
        original_shape = x.shape
        x_reshaped = x.reshape(original_shape[0], -1)  # [C, H*W]
        rows, cols = x_reshaped.shape
    
    # Choose appropriate block sizes for GPU
    BLOCK_SIZE_M = 32    # Process multiple rows in parallel
    BLOCK_SIZE_N = 1024  # Process columns in large blocks
    
    # Calculate grid dimensions
    num_rows = (rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_cols = (cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(x_reshaped)
    
    # Launch fused kernel
    softmax_dropout_kernel[(num_rows, num_cols)](
        x_ptr=x_reshaped,
        out_ptr=out,
        rows=rows,
        cols=cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape back to original dimensions
    if len(original_shape) == 4:
        return out.reshape(original_shape[1], original_shape[2], original_shape[3])
    else:
        return out.reshape(original_shape[0], original_shape[1], original_shape[2])

# Argument extraction function
def replacement_args(x):
    return (x,)

# Replacement function - returns the fused function
def replacement_func():
    return fused_softmax_dropout