import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple split pattern to test basic matching"""
    split = x.split([32, 32, 128], dim=3)
    return split

def replacement_args(x):
    """Extract arguments needed for the replacement kernel"""
    return (x,)

@triton.jit
def split_kernel(
    input_ptr,
    output0_ptr,
    output1_ptr,
    output2_ptr,
    batch_size,
    height,
    width,
    total_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that splits tensor into 3 parts along dim=3"""
    pid = tl.program_id(0)
    
    # Each thread processes one element
    elem_idx = pid
    
    # Calculate 4D coordinates from flat index
    batch_idx = elem_idx // (height * width * total_channels)
    remaining = elem_idx % (height * width * total_channels)
    h_idx = remaining // (width * total_channels)
    remaining = remaining % (width * total_channels)
    w_idx = remaining // total_channels
    c_idx = remaining % total_channels
    
    # Skip if out of bounds
    if batch_idx >= batch_size or h_idx >= height or w_idx >= width or c_idx >= total_channels:
        return
    
    # Calculate input coordinate
    input_coord = batch_idx * height * width * total_channels + h_idx * width * total_channels + w_idx * total_channels + c_idx
    input_val = tl.load(input_ptr + input_coord)
    
    # Determine which output tensor and position based on channel dimension
    # Split sizes: [32, 32, 128]
    if c_idx < 32:
        # First split: goes to output0
        out_coord = batch_idx * height * width * 32 + h_idx * width * 32 + w_idx * 32 + c_idx
        tl.store(output0_ptr + out_coord, input_val)
    elif c_idx < 64:
        # Second split: goes to output1  
        out_coord = batch_idx * height * width * 32 + h_idx * width * 32 + w_idx * 32 + (c_idx - 32)
        tl.store(output1_ptr + out_coord, input_val)
    else:
        # Third split: goes to output2
        out_coord = batch_idx * height * width * 128 + h_idx * width * 128 + w_idx * 128 + (c_idx - 64)
        tl.store(output2_ptr + out_coord, input_val)

@torch.fx.wrap
def split_replacement(x):
    """Replacement that implements split using Triton kernel"""
    batch_size, height, width, total_channels = x.shape
    
    # Create output tensors using allowed methods
    # Note: the split operation permutes dimensions, so output shapes change
    output0 = torch.empty((batch_size, height, width, 32), dtype=x.dtype, device=x.device)
    output1 = torch.empty((batch_size, height, width, 32), dtype=x.dtype, device=x.device)
    output2 = torch.empty((batch_size, height, width, 128), dtype=x.dtype, device=x.device)
    
    # Total number of elements in the tensor
    total_elements = batch_size * height * width * total_channels
    grid = (total_elements,)
    
    split_kernel[grid](
        x,
        output0, 
        output1,
        output2,
        batch_size,
        height,
        width,
        total_channels,
        1,  # BLOCK_SIZE is not used in this simple approach
    )
    
    return output0, output1, output2

def replacement_func():
    """Returns the split replacement function"""
    return split_replacement