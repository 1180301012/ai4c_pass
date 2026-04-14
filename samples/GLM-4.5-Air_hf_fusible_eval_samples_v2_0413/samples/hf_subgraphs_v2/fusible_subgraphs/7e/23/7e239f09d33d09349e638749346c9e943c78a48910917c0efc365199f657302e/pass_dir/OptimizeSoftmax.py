import torch
import triton
import triton.language as tl
import math

def pattern(in_1):
    # Match the softmax operation
    result = in_1.softmax(dim=-1)
    return result

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def optimized_softmax_kernel(x_ptr, out_ptr, n_cols, n_rows, stride, BLOCK_SIZE: tl.constexpr):
    """
    Optimized softmax kernel for large rectangular matrices
    Args:
        x_ptr: pointer to input tensor
        out_ptr: pointer to output tensor  
        n_cols: number of columns (inner dimension)
        n_rows: number of rows (outer dimension)
        stride: stride between rows
        BLOCK_SIZE: number of elements processed per thread
    """
    # Each program handles one row
    row_id = tl.program_id(0)
    row_start = row_id * stride + tl.arange(0, BLOCK_SIZE)
    
    # Row-wise reduction for max
    max_val = -float('inf')
    offset = 0
    while offset < n_cols:
        current_block = min(BLOCK_SIZE, n_cols - offset)
        mask = tl.arange(0, current_block) < current_block
        
        # Load a block of the row
        x_block = tl.load(x_ptr + row_start, mask=mask, other=-float('inf'))
        max_val = tl.max(max_val, x_block)
        
        # Advance by block size (in elements, not bytes)
        row_start += BLOCK_SIZE
        offset += BLOCK_SIZE
    
    # All-reduce for max within the row
    max_val = tl.max(max_val, 0)  # Single element reduction
    
    # Compute row-wise sum of exp(x - max)
    sum_exp = 0.0
    offset = 0
    row_ptr = row_id * stride
    
    while offset < n_cols:
        current_block = min(BLOCK_SIZE, n_cols - offset)
        mask = tl.arange(0, current_block) < current_block
        
        # Load a block of the row
        x_block = tl.load(x_ptr + row_ptr + offset, mask=mask, other=-float('inf'))
        
        # Compute exp(x - max) with stability
        exp_x = tl.exp(x_block - max_val)
        sum_exp += tl.sum(exp_x)
        
        offset += BLOCK_SIZE
    
    # Final normalization and store
    inv_sum = 1.0 / sum_exp
    offset = 0
    
    while offset < n_cols:
        current_block = min(BLOCK_SIZE, n_cols - offset)
        mask = tl.arange(0, current_block) < current_block
        
        # Load and compute normalized softmax
        x_block = tl.load(x_ptr + row_ptr + offset, mask=mask, other=-float('inf'))
        softmax_val = tl.exp(x_block - max_val) * inv_sum
        
        # Store result
        tl.store(out_ptr + row_ptr + offset, softmax_val, mask=mask)
        
        offset += BLOCK_SIZE

@triton.jit
def efficient_softmax_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    More efficient softmax implementation that handles blocks better
    """
    # Each program handles one row
    pid = tl.program_id(0)
    
    # Load the row (row-wise operation)
    offsets = tl.arange(0, N)
    mask = offsets < N
    
    # Load entire row (inner dimension)
    x_row = tl.load(x_ptr + pid * N + offsets, mask=mask, other=-float('inf'))
    
    # Find max in row
    row_max = tl.max(x_row)
    
    # Compute exp(x - max) and sum
    exp_x = tl.exp(x_row - row_max)
    row_sum = tl.sum(exp_x)
    
    # Normalize
    softmax_row = exp_x / row_sum
    
    # Store results
    tl.store(out_ptr + pid * N + offsets, softmax_row, mask=mask)

@torch.fx.wrap
def optimized_softmax(input_tensor):
    """High-performance softmax for large rectangular matrices"""
    if input_tensor.dim() != 3:
        # Fallback for non-3D tensors
        return input_tensor.softmax(dim=-1)
        
    B, H, W = input_tensor.shape
    
    # Use different strategies based on dimensions
    if W <= 4096:  # Inner dimension is manageable
        # Use simple row-wise softmax
        N = input_tensor.numel() // B  # Elements per row
        BLOCK_SIZE = 1024
        num_rows = B * H  # Treat as flattened rows
        
        output = torch.empty_like(input_tensor)
        
        efficient_softmax_kernel[(num_rows,)](
            x_ptr=input_tensor,
            out_ptr=output,
            M=num_rows,
            N=N,
            BLOCK_M=1,
            BLOCK_N=1024
        )
        
        return output
    else:
        # For very large dimensions, use a more complex approach
        return input_tensor.softmax(dim=-1)

def replacement_func():
    return optimized_softmax