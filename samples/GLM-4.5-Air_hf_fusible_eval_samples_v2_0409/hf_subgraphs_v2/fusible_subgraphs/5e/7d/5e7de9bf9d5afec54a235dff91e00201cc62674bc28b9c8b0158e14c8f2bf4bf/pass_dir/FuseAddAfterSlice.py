import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Pattern to match the slice + add operation:
    a[(Ellipsis, slice(None, 124, None))] + b[(Ellipsis, slice(None, 124, None))] 
    """
    slice_a = a[(Ellipsis, slice(None, 124, None))]
    slice_b = b[(Ellipsis, slice(None, 124, None))]
    result = slice_a + slice_b
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def fused_slice_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, end_idx, BLOCK_SIZE: tl.constexpr):
    """Fused kernel that slices and adds tensors in one operation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle the actual slice (first 124 elements) and bounds checking
    mask = (offsets < end_idx) & (offsets < BLOCK_SIZE + (pid * BLOCK_SIZE))
    
    # Load elements up to the slice boundary
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform fused addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_slice_add(a, b):
    """Wrapper function to launch the fused slice-add kernel"""
    # Get slice dimensions from the original pattern
    slice_end = 124  # Hardcoded based on the original slice operation
    
    # Determine the actual number of elements to process
    n_elements = min(slice_end, a.shape[-1])
    
    # Create output tensor
    out_shape = list(a.shape[:-1]) + [slice_end]
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # Launch kernel with appropriate grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_slice_add_kernel[(num_programs,)](
        x_ptr=a,
        y_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        end_idx=slice_end,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the sliced and added result (matching the original pattern output)
    return out

def replacement_func():
    return fused_slice_add