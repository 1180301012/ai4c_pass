import torch
import triton
import triton.language as tl

# Pattern matching function for meshgrid + flatten operations
def pattern(in_1, in_0):
    # Match the exact pattern from the computation:
    # arange -> meshgrid -> flatten both outputs
    tmp_1 = torch.arange(8, dtype=torch.float32).to(in_1.device)
    tmp_2 = torch.meshgrid(in_1, tmp_1, indexing='ij')
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_3.flatten()
    tmp_6 = tmp_4.flatten()
    return tmp_5, tmp_6

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Optimized kernel for direct coordinate generation
@triton.jit
def meshgrid_flatten_kernel(
    in_1_ptr,        # Input tensor pointer  
    out_x_ptr,       # Output x coordinates pointer
    out_y_ptr,       # Output y coordinates pointer
    in_1_size,       # Size of input tensor
    arange_size,     # Size of arange tensor (fixed to 8)
    total_size,      # Total flattened size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    
    # Calculate original 2D coordinates from flattened index
    i = offsets // arange_size  # Row index (0-7)
    j = offsets % arange_size   # Column index (0-7)
    
    # Load corresponding 1D values
    # For x coordinates: use in_1 at column index j
    x_val = tl.load(in_1_ptr + j, mask=mask, other=0.0)
    # For y coordinates: use arange values at row index i
    y_val = tl.float32(i)  # arange creates 0,1,2,3,4,5,6,7
    
    # Store flattened coordinates
    tl.store(out_x_ptr + offsets, x_val, mask=mask)
    tl.store(out_y_ptr + offsets, y_val, mask=mask)

# Kernel wrapper
@torch.fx.wrap  
def optimized_meshgrid_flatten(x, y):
    # The arange tensor is created inside the kernel now
    # We only need x which is in_1 (the input tensor)
    arange_size = 8  # Fixed size
    x_size = x.size(0)
    y_size = arange_size
    total_size = x_size * y_size
    
    # Create output tensors
    out_x = torch.empty(total_size, dtype=x.dtype, device=x.device)
    out_y = torch.empty(total_size, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    meshgrid_flatten_kernel[(num_programs,)](
        in_1_ptr=x,
        out_x_ptr=out_x,
        out_y_ptr=out_y,
        in_1_size=x_size,
        arange_size=y_size,
        total_size=total_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_x, out_y

# Replacement function
def replacement_func():
    return optimized_meshgrid_flatten