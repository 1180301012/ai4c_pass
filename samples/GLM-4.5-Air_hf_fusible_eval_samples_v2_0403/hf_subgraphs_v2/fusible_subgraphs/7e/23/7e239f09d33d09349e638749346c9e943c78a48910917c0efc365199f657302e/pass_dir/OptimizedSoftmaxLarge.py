import torch
import triton
import triton.language as tl
import math

def pattern(x):
    result = x.softmax(dim=-1)
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_softmax_kernel(
    x_ptr, 
    out_ptr, 
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the matrix
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for the columns within this block
    mask = col_offsets < n_cols
    
    # Load one row (contiguous within the block)
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    x = tl.where(mask, x, -float('inf'))
    
    # Find max for numerical stability
    max_val = tl.max(x)
    
    # Subtract max and exponentiate
    shifted_x = x - max_val
    exp_x = tl.exp(shifted_x)
    
    # Compute sum
    sum_exp = tl.sum(exp_x)
    
    if sum_exp == 0:
        sum_exp = 1.0
    
    # Normalize
    softmax_exp = exp_x / sum_exp
    softmax_exp = tl.where(mask, softmax_exp, 0.0)
    
    # Store result
    tl.store(out_ptr + row_idx * n_cols + col_offsets, softmax_exp, mask=mask)

@torch.fx.wrap
def optimized_softmax_large(x):
    """
    Optimized softmax for large tensors with efficient memory access and GPU utilization.
    Handles tensors with shape [B, ..., N] and applies softmax on last dimension (dim=-1).
    """
    # Ensure the tensor is on CUDA
    if x.device.type != 'cuda':
        x = x.cuda()
    
    # For our use case, we're dealing with 3D tensors [batch, rows, cols]
    # We want to apply softmax on the last dimension (cols)
    if x.dim() == 3:
        batch, rows, cols = x.shape
        n_rows = batch * rows
        n_cols = cols
        x_reshaped = x.reshape(n_rows, cols)
    else:
        # Fallback to original shape handling
        original_shape = x.shape
        n_rows = x.numel() // x.shape[-1] if x.numel() > 0 else 1
        n_cols = x.shape[-1] if x.dim() > 0 else 1
        x_reshaped = x.reshape(n_rows, n_cols)
    
    # Create output tensor with the correct dtype
    if x.dtype == torch.bfloat16:
        out_dtype = torch.bfloat16
    elif x.dtype == torch.float16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32
    
    out = torch.empty_like(x_reshaped, dtype=out_dtype)
    
    # Optimize block size based on tensor size
    BLOCK_SIZE = 2048  # Use a larger block size for better GPU utilization
    
    # Number of programs needed (each program handles one row, processes BLOCK_SIZE columns)
    n_blocks_n = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_softmax_kernel[(n_rows, n_blocks_n)](
        x_ptr=x_reshaped,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original dimensions
    if x.dim() == 3:
        # Reshape back to original 3D form [batch, rows, cols]
        batch, rows, cols = x.shape
        out = out.reshape(batch, rows, cols)
    else:
        # For other dimensions, keep as reshaped
        pass
    
    return out

def replacement_func():
    return optimized_softmax_large