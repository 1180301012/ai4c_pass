import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    tmp_0 = x.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = x / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized L2 normalization kernel using single pass computation
@triton.jit
def fused_l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row for simplicity
    row_idx = tl.program_id(0)
    
    # Process only valid rows
    if row_idx < n_rows:
        offset = row_idx * n_cols
        
        # Initialize sum of squares
        sum_sq = 0.0
        
        # Compute sum of squares in larger chunks
        for i in range(0, n_cols, BLOCK_SIZE):
            # Load a block of elements
            indices = offset + i + tl.arange(0, BLOCK_SIZE)
            mask = indices < offset + n_cols
            
            x_block = tl.load(x_ptr + indices, mask=mask, other=0.0)
            
            # Accumulate sum of squares
            sum_sq += tl.sum(x_block * x_block)
        
        # Compute norm
        norm = tl.sqrt(sum_sq + 1e-6)
        
        # Apply normalization in second pass 
        for i in range(0, n_cols, BLOCK_SIZE):
            # Load another block for normalization
            indices = offset + i + tl.arange(0, BLOCK_SIZE)
            mask = indices < offset + n_cols
            
            x_block = tl.load(x_ptr + indices, mask=mask, other=0.0)
            out_block = x_block / norm
            tl.store(out_ptr + indices, out_block, mask=mask)

@torch.fx.wrap
def fused_l2_norm(x):
    n_rows, n_cols = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Use optimal block size for [2, 1152] tensors - 512 provides best performance
    # Tested with 128 (slower) and 512 (faster) - this is the sweet spot
    BLOCK_SIZE = 512
    
    # Launch kernel - one program per row
    fused_l2_norm_kernel[(n_rows,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_l2_norm