import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Match the tensor expansion pattern:
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    """
    # This is adding two None dimensions at the front and keeping the original dim
    expanded = in_0[(None, None, slice(None, None, None))]
    return expanded

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel for tensor expansion
@triton.jit
def tensor_expansion_kernel(
    in_ptr,
    out_ptr,
    in_height,
    in_width,
    out_batch,
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one output element
    batch_idx = tl.program_id(0)
    height_idx = tl.program_id(1) 
    width_idx = tl.program_id(2)
    
    # Calculate global output offset
    global_offset = batch_idx * out_height * out_width + height_idx * out_width + width_idx
    
    # For expanded tensor [1, 1, in_height, in_width], 
    # any access with batch_idx=0 and height_idx=0 maps to in_0[0, height_idx, width_idx]
    if batch_idx == 0 and height_idx < in_height and width_idx < in_width:
        in_offset = height_idx * in_width + width_idx
        # Load from input and store to output
        tl.store(out_ptr + global_offset, tl.load(in_ptr + in_offset))
    else:
        # Fill with zeros for out of bounds elements (but in this case we only expand to 1,1)
        tl.store(out_ptr + global_offset, 0.0)

# Optimized kernel using view-based approach
@triton.jit  
def tensor_expansion_view_kernel(
    in_ptr, 
    out_ptr,
    in_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < in_elements
    
    # Load input data directly
    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Store to output at corresponding positions
    # The expansion [None, None, slice(...)] creates a [1, 1, H, W] tensor
    out_offsets = offsets  # mapping 1:1 since extra dims are size 1
    tl.store(out_ptr + out_offsets, data, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_tensor_expansion(in_0):
    # Get input dimensions
    in_height, in_width = in_0.shape
    in_elements = in_height * in_width
    
    # The expansion [None, None, slice(...)] creates [1, 1, in_height, in_width]
    out_shape = (1, 1, in_height, in_width)
    out_elements = in_elements  # Same number of elements, just different shape
    
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate optimal block size
    BLOCK_SIZE = min(1024, out_elements)
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use the simpler view-based kernel
    tensor_expansion_view_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        in_elements=out_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_tensor_expansion