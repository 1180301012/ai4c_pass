import torch
import triton
import triton.language as tl

def pattern(in_1):
    """
    Pattern matching: view operation that reshapes the second input
    This mirrors the computation from model.py:
    tmp_5 = in_1.view(batches, channels, -1)
    
    Note: The exact view parameters vary across graphs, but they all follow
    the pattern of reshaping [batches, channels, height, width] to 
    [batches, channels, height*width]
    """
    # We need to determine the target shape dynamically based on input
    # Since all graphs view to [batches, channels, spatial] where spatial = height * width
    batches, channels, height, width = in_1.shape
    spatial = height * width
    tmp_5 = in_1.view(batches, channels, spatial)
    return tmp_5

def replacement_args(in_1):
    """Extract arguments for replacement function"""
    return (in_1,)

@triton.jit
def optimized_view_kernel(
    input_ptr,
    output_ptr,
    batches,
    channels,
    height, 
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for tensor reshaping from [batches, channels, height, width] 
    to [batches, channels, height * width]
    
    Uses optimized memory access patterns with 1D grid for better GPU occupancy
    """
    # Compute program ID for 1D grid
    pid = tl.program_id(0)
    
    # Calculate total number of elements (spatial dimensions only)
    spatial_elements = height * width
    total_elements = batches * channels * spatial_elements
    
    # Check if this program is within bounds
    if pid >= total_elements:
        return
    
    # Convert 1D element ID to batch, channel, and spatial coordinates
    spatial_idx = pid % spatial_elements
    channel_idx = (pid // spatial_elements) % channels
    batch_idx = pid // (channels * spatial_elements)
    
    # Convert 1D spatial index to 2D (height, width) coordinates
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Calculate input and output pointers
    input_offset = (batch_idx * channels + channel_idx) * height * width + h_idx * width + w_idx
    output_offset = pid
    
    # Load from input 4D layout and store to output 3D layout
    val = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap  
def optimized_view_func(x):
    """Function that applies optimized view operation"""
    # Get input shape
    batches, channels, height, width = x.shape
    spatial = height * width
    total_elements = batches * channels * spatial
    
    # Use adaptive block size based on total elements
    if total_elements <= 16384:  # 4K elements
        BLOCK_SIZE = 256
    elif total_elements <= 65536:  # 64K elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Determine number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((batches, channels, spatial), dtype=x.dtype, device=x.device)
    
    # Launch kernel with 1D grid for better GPU occupancy
    optimized_view_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        batches=batches,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_view_func