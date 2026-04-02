import torch
import triton
import triton.language as tl

def pattern(x):
    s = x.sum(dim=-1)
    result = x / s.unsqueeze(-1)
    return (result,)

def replacement_args(x):
    return (x,)

@triton.jit
def normalize_last_dim_kernel(
    x_ptr,
    out_ptr,
    total_elements,
    last_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the flattened tensor
    pid = tl.program_id(0)
    
    if pid >= total_elements:
        return
        
    # Calculate row index (each row spans last_dim_size elements)
    row_idx = pid // last_dim_size
    
    # Calculate start index of this row
    row_start = row_idx * last_dim_size
    
    # Create mask for all elements in this row
    row_mask = row_start + tl.arange(0, last_dim_size) < row_start + last_dim_size
    
    # Load the entire row to compute the sum
    row = tl.load(x_ptr + row_start, mask=row_mask, other=0.0)
    row_sum = tl.sum(row)
    
    # Normalize the current element
    current_val = tl.load(x_ptr + pid)
    normalized_val = current_val / row_sum
    
    # Store the result
    tl.store(out_ptr + pid, normalized_val)

@torch.fx.wrap
def fused_normalization(x):
    # Get input shape
    batch_size, heads, seq_len, last_dim = x.shape
    total_elements = x.numel()
    
    # Set up Triton kernel configuration - each program handles one element for simplicity
    num_programs = total_elements
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel - each program processes one element
    normalize_last_dim_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        total_elements=total_elements,
        last_dim_size=last_dim,
        BLOCK_SIZE=1,  # Each program handles one element
    )
    
    return out

def replacement_func():
    return fused_normalization