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
    
    # Process a contiguous block of elements
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Calculate input indices: [batch, channels, h, w]
        batch_idx = idx // (channels * height * width)
        remaining = idx % (channels * height * width)
        channel_idx = remaining // (height * width)
        spatial_idx = remaining % (height * width)
        h_idx = spatial_idx // width
        w_idx = spatial_idx % width
        
        # Calculate output indices after flatten+transpose: [batch, h*w, channels]
        new_idx = batch_idx * (channels * height * width) + \
                 (h_idx * width + w_idx) * channels + channel_idx
        
        # Direct memory copy with optimized memory access pattern
        if idx < batch_size * channels * height * width and new_idx < batch_size * channels * height * width:
            val = tl.load(input_ptr + idx)
            tl.store(output_ptr + new_idx, val)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    total_elements = batch_size * channels * height * width
    
    # Optimized block size for 128 channels and various spatial dimensions
    if height * width >= 10000:  # Very large spatial dimensions (96x96 = 9216)
        BLOCK_SIZE = 4096
    elif height * width >= 2500:  # Medium spatial dimensions
        BLOCK_SIZE = 8192
    else:  # Small spatial dimensions
        BLOCK_SIZE = 16384
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_tensor = torch.empty((batch_size, height * width, channels), 
                              dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with optimal configuration for 128 channels
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