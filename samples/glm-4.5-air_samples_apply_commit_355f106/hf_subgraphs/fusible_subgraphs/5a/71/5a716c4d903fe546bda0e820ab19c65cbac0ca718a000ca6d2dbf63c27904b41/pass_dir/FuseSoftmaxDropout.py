import torch
import triton
import triton.language as tl
import math

def pattern(tmp_3):
    # Fuse: dropout(softmax(tmp_3, dim=-1), p=0.1, training=False)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
    return tmp_5

def replacement_args(tmp_3):
    return (tmp_3,)

@triton.jit
def softmax_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    last_dim_size: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Each program handles a block of the tensor
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Calculate offset for this program
    row_offset = row_idx * BLOCK_ROWS * last_dim_size
    
    # Process columns in blocks
    block_start = col_idx * BLOCK_SIZE
    block_end = min((col_idx + 1) * BLOCK_SIZE, last_dim_size)
    
    # Create mask for valid elements
    mask = (block_start + tl.arange(0, block_end - block_start)) < last_dim_size
    
    if mask.numel() == 0:
        return
    
    # Load the current row slice
    row_ptr = x_ptr + row_offset + block_start
    x_row = tl.load(row_ptr + tl.arange(0, block_end - block_start), mask=mask)
    
    # Compute max for numerical stability
    max_val = tl.max(x_row, mask=mask)
    
    # Compute exponentials
    exp_x = tl.exp(x_row - max_val)
    
    # Compute sum for softmax denominator
    sum_exp = tl.sum(exp_x, mask=mask)
    
    # Compute softmax (during inference, dropout is just scaling)
    # When training=False, dropout is just multiplying by (1 - dropout_p)
    scale_factor = 1.0 - dropout_p
    softmax_out = exp_x * scale_factor / sum_exp
    
    # Store result
    out_row_ptr = out_ptr + row_offset + block_start
    tl.store(out_row_ptr + tl.arange(0, block_end - block_start), softmax_out, mask=mask)

@torch.fx.wrap  
def fused_softmax_dropout(tmp_3, dropout_p=0.1, training=False):
    # Only handle 4D tensors (the expected shape for attention)
    input_shape = tmp_3.shape
    
    # For unsupported shapes, reshape to 4D
    if len(input_shape) != 4:
        # Reshape to 4D by adding singleton dimensions
        while len(input_shape) < 4:
            tmp_3 = tmp_3.unsqueeze(1)
            input_shape = tmp_3.shape
    
    N, H, L, S = input_shape
    last_dim_size = S
    n_rows = N * H * L
    
    # Create output tensor
    out = torch.empty_like(tmp_3)
    
    # Launch 2D grid: rows and columns
    BLOCK_SIZE = 1024  # Tune this for optimal performance
    BLOCK_ROWS = 1  # Process one row at a time for softmax stability
    
    # Calculate grid dimensions
    n_cols = (last_dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_rows, n_cols)
    
    # Launch kernel
    softmax_dropout_kernel[grid](
        tmp_3,
        out,
        tmp_3.numel(),
        last_dim_size=last_dim_size,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_ROWS=BLOCK_ROWS,
    )
    
    # If we added dimensions, remove them
    if len(input_shape) != 4:
        # Remove the singleton dimensions we added
        while len(out.shape) > len(input_shape):
            out = out.squeeze(1)
    
    return out

def replacement_func():
    return fused_softmax_dropout