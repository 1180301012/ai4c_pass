import torch
import triton
import triton.language as tl

def pattern(x):
    """Match sum + unsqueeze + division pattern after dropout elimination"""
    tmp_0 = x.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_0 = None
    x /= tmp_1
    tmp_2 = x
    tmp_1 = None
    return (tmp_2,)

def replacement_args(x):
    """Extract arguments - we only need the input tensor x"""
    return (x,)

@triton.jit
def sum_div_kernel(
    x_ptr, 
    out_ptr, 
    n_cols,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: compute sum along last dim and apply normalization"""
    # Program ID
    pid = tl.program_id(0)
    
    # Each program handles one element in all dimensions except the last
    # For shape [1, 16, 196, 196], each program handles one row of [196]
    row_offset = pid * n_cols
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum for this row
    row_sum = tl.sum(x, mask=mask)
    
    # Normalize: divide each element by the sum (add small epsilon for stability)
    epsilon = 1e-7
    out = x / (row_sum + epsilon)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def sum_div_norm(x):
    """Normalize each row by its sum along the last dimension"""
    # For input shape [1, 16, 196, 196]
    # We want to normalize along the last dimension (196)
    
    # Get total number of elements
    n_elements = x.numel()
    # Get size of last dimension (196)
    n_cols = x.shape[-1]
    # Number of programs (total elements / elements per row)
    n_rows = n_elements // n_cols
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Block size should match the last dimension size for optimal performance
    BLOCK_SIZE = n_cols
    
    # Kernel launch
    sum_div_kernel[(n_rows,)](
        x_ptr=x,
        out_ptr=out,
        n_cols=n_cols,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused sum+div normalization function as a tuple"""
    def wrapped_sum_div_norm(x):
        result = sum_div_norm(x)
        return (result,)  # Return as tuple to match pattern output format
    return wrapped_sum_div_norm