import torch
import triton
import triton.language as tl

def pattern(in_tensor):
    cos_result = in_tensor.cos()
    sin_result = in_tensor.sin()
    concat_result = torch.cat((cos_result, sin_result), dim=-1)
    return concat_result

def replacement_args(in_tensor):
    return (in_tensor,)

@triton.jit
def fused_cos_sin_concat_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row with 1D grid
    row_idx = tl.program_id(0)
    
    # Only process if within bounds
    if row_idx >= n_rows:
        return
    
    # Base addresses for this row
    in_base = row_idx * n_cols
    out_base = row_idx * (n_cols * 2)
    
    # Process elements in this row
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load inputs
        x = tl.load(in_ptr + in_base + offsets, mask=mask, other=0.0)
        
        # Compute cos and sin
        cos_vals = tl.cos(x)
        sin_vals = tl.sin(x)
        
        # Store cos in first half
        tl.store(out_ptr + out_base + offsets, cos_vals, mask=mask)
        
        # Store sin in second half  
        tl.store(out_ptr + out_base + offsets + n_cols, sin_vals, mask=mask)

@torch.fx.wrap
def fused_cos_sin_concat(in_tensor):
    n_rows = in_tensor.shape[0]
    n_cols = in_tensor.shape[1]
    
    # Use 1D grid where each program handles one row
    num_programs = n_rows
    
    # Use larger block size for better performance
    BLOCK_SIZE = 256  # Process 256 elements per thread group
    
    # Output shape: [n_rows, 2*n_cols]
    out_shape = [n_rows, 2 * n_cols]
    out = torch.empty(out_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    fused_cos_sin_concat_kernel[(num_programs,)](
        in_ptr=in_tensor,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_cos_sin_concat