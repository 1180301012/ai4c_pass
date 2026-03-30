import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern to match: redundant torch.cat + normalize operation
    The concat operation torch.cat([in_0], 1) is redundant since concatenating 
    a tensor with itself returns the same tensor.
    """
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return (tmp_1,)  # Return tuple to match the expected pattern

def replacement_args(in_0):
    """Extract arguments for the replacement function"""
    return (in_0,)

@triton.jit
def l2_normalize_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance L2 normalization kernel using Triton.
    Normalizes each row of the input tensor (dim=1).
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate row offset in memory
    row_offset = row_idx * n_cols
    
    # Load current row data
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load the row
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute L2 norm of the row
    squared_sum = tl.sum(x * x)
    norm = tl.sqrt(squared_sum)
    
    # Handle zero norm case to avoid division by zero
    norm = tl.where(norm == 0.0, 1.0, norm)
    
    # Normalize the row
    out = x / norm
    
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def custom_l2_normalize(input_tensor):
    """
    Custom L2 normalization function that operates along dimension 1
    """
    n_rows, n_cols = input_tensor.shape
    
    # Choose optimal block size based on feature dimension
    if n_cols >= 1024:
        BLOCK_SIZE = 1024
    elif n_cols >= 512:
        BLOCK_SIZE = 512
    elif n_cols >= 256:
        BLOCK_SIZE = 256
    elif n_cols >= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 64
    
    # Calculate number of programs needed (one per row)
    num_programs = n_rows
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch the kernel
    l2_normalize_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return custom_l2_normalize