import torch
import triton
import triton.language as tl

@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Work-items within each row
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid column indices in this row
    mask = offsets < n_cols
    
    # Load the current row
    x_ptr_row = x_ptr + row_idx * n_cols
    x = tl.load(x_ptr_row + offsets, mask=mask, other=0.0)
    
    # Compute L2 norm for the row
    sum_sq = tl.sum(x * x)
    norm = tl.sqrt(sum_sq)
    
    # Store the norm (each row gets its own output position)
    tl.store(norm_ptr + row_idx, norm)
    
    # Normalize the row
    out = x / norm
    tl.store(out_ptr + row_idx * n_cols + offsets, out, mask=mask)

@torch.fx.wrap
def normalize_tensor_triton(x):
    # Input shape: [n_rows, n_cols]
    n_rows, n_cols = x.shape
    n_elements = n_rows * n_cols
    
    # Set block size - should be multiple of 32 for GPU efficiency
    BLOCK_SIZE = 1024
    num_programs = n_rows  # One program per row
    
    # Output tensor - return only the normalized result
    out = torch.empty_like(x)
    
    # For the pattern matching, we only need to return the normalized tensor
    # Create a dummy norm output that we won't actually use in the pattern
    norm_out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    # Launch kernel for normalization
    normalize_kernel[(num_programs,)](
        x_ptr=x,
        norm_ptr=norm_out,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out  # Only return the normalized tensor to match pattern



def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

def replacement_func():
    return normalize_tensor_triton