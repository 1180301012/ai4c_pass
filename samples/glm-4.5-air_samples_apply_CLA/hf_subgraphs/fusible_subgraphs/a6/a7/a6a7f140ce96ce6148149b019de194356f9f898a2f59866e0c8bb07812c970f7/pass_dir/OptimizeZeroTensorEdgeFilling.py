import torch
import triton
import triton.language as tl

# Pattern matching for simple tensor operations  
def pattern(tmp_0):
    # Simple pattern: match fill_ operation with slice
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    return tmp_0, tmp_2

# Argument extraction function
def replacement_args(tmp_0):
    return (tmp_0,)

# Optimized kernel for creating tensor with edge ones directly
@triton.jit
def create_edge_ones_kernel(
    out_ptr,
    n_rows,
    n_cols,
    edge_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate grid position
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Each thread handles a block of columns
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    row_ptr = out_ptr + row * n_cols
    
    # Check if we're in the edge region
    if row < edge_size or row >= n_rows - edge_size:
        # Top or bottom edge - fill entire row with 1
        tl.store(row_ptr + offsets, 1.0, mask=mask)
    else:
        # Middle rows - only fill edge columns
        if col < edge_size or col >= n_cols - edge_size:
            tl.store(row_ptr + offsets, 1.0, mask=mask)
        else:
            tl.store(row_ptr + offsets, 0.0, mask=mask)

@torch.fx.wrap
def create_tensor_with_edge_ones(shape, device, edge_size=5):
    n_rows, n_cols = shape[1], shape[2]
    out = torch.zeros(shape, device=device, dtype=torch.float32)
    
    BLOCK_SIZE = 32
    grid = (
        (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    
    create_edge_ones_kernel[grid](
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        edge_size=edge_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Extract edge slice from the tensor
def get_edge_slice(tensor):
    # Get slice from last 5 rows
    slice_rows = tensor[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    return slice_rows

@torch.fx.wrap
def optimized_tensor_creation(shape, device, edge_size=5):
    # Create tensor with edge ones directly
    tensor_with_edges = create_tensor_with_edge_ones(shape, device, edge_size)
    
    # Get the required slice
    slice_result = get_edge_slice(tensor_with_edges)
    
    return tensor_with_edges, slice_result

# Replacement function
def replacement_func():
    return optimized_tensor_creation