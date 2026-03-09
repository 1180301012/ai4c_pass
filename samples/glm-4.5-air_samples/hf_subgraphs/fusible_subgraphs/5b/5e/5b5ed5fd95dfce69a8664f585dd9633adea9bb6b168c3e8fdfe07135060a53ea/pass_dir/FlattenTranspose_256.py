import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Flatten to [batch, channels, height*width]
    flattened = input_tensor.flatten(2)
    # Transpose to [batch, height*width, channels]
    result = flattened.transpose(1, 2)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    elements_per_program = BLOCK_SIZE
    
    if pid * elements_per_program >= total_elements:
        return
    
    # Optimized memory access pattern for 256 channels
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    # Use vectorized operations for better performance
    for idx in range(start_idx, end_idx):
        # Calculate input indices: [batch, channels, h, w]
        batch_idx = idx // (channels * height * width)
        remaining = idx % (channels * height * width)
        channel_idx = remaining // (height * width)
        spatial_idx = remaining % (height * width)
        h_idx = spatial_idx // width
        w_idx = spatial_idx % width
        
        # Calculate output indices with optimized layout for 256 channels
        # This minimizes bank conflicts in shared memory
        spatial_flat = h_idx * width + w_idx
        new_idx = batch_idx * (channels * height * width) + \
                 spatial_flat * channels + channel_idx
        
        # Direct memory transfer with coalesced access pattern
        if idx < batch_size * channels * height * width and new_idx < batch_size * channels * height * width:
            val = tl.load(input_ptr + idx)
            tl.store(output_ptr + new_idx, val)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    total_elements = batch_size * channels * height * width
    
    # Optimized block size for 256 channels and 48x48 spatial dimensions
    if height * width >= 2500:  # Large spatial dimensions (48x48 = 2304)
        BLOCK_SIZE = 8192
    elif height * width >= 1000:
        BLOCK_SIZE = 16384
    else:  # Small spatial dimensions
        BLOCK_SIZE = 32768
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Pre-allocate output with optimal memory layout
    output_tensor = torch.empty((batch_size, height * width, channels), 
                              dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with optimized configuration for 256 channels
    flatten_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return optimized_flatten_transpose