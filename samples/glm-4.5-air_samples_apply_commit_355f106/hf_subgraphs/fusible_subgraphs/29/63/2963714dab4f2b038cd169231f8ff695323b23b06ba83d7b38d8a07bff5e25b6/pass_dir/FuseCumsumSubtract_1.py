import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Original pattern: cumsum followed by subtract 1
    tmp_1 = x.cumsum(-1)
    tmp_2 = tmp_1 - 1
    return tmp_2

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_cumsum_subtract_kernel(
    x_ptr,
    out_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the tensor
    pid = tl.program_id(0)
    
    # Calculate offset for this row
    row_start = pid * stride
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input data for this row
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Perform fused cumsum and subtract-1 for the row
    # This assumes the input is 2D and we process each row independently
    if x.shape[0] > 0:
        # First element in row
        first_val = x[0] if mask[0] else 0
        tl.store(out_ptr + offsets[0], first_val - 1, mask=mask[0])
        
        # Cumsum for remaining elements in the row
        for i in range(1, BLOCK_SIZE):
            if mask[i]:
                prev_val = tl.load(out_ptr + offsets[i-1])
                current_val = prev_val + x[i]
                tl.store(out_ptr + offsets[i], current_val - 1, mask=mask[i])
    
    # For inter-row carry-over (if needed for 1D cumsum across all elements)
    # This version handles 1D cumsum by treating it as multiple 1D operations

@torch.fx.wrap  
def fused_cumsum_subtract(x, y):
    # Input x can be 1D or 2D
    if x.dim() == 1:
        N = x.numel()
        BLOCK_SIZE = min(1024, N)
        num_rows = 1
        stride = N
    else:
        # For 2D, assume we want cumsum along last dimension independently for each row
        N = x.numel() // x.shape[0]  # elements per row
        BLOCK_SIZE = min(1024, N)
        num_rows = x.shape[0]
        stride = N
    
    out = torch.empty_like(x)
    num_programs = num_rows
    
    # For 1D case with proper inter-block carry-over
    if x.dim() == 1:
        # Use simple Python loop for cumsum - this is small tensors based on input metadata
        result = torch.empty_like(x)
        if x.numel() > 0:
            result[0] = x[0] - 1
            for i in range(1, x.numel()):
                result[i] = result[i-1] + x[i]
        out.copy_(result)
    else:
        # For 2D case, we can process each row independently
        fused_cumsum_subtract_kernel[(num_rows,)](
            x_ptr=x,
            out_ptr=out,
            stride=stride,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return fused_cumsum_subtract