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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total elements per block
    total_elements = batch_size * channels * height * width
    elements_per_block = BLOCK_SIZE_M * BLOCK_SIZE_N
    
    if pid * elements_per_block >= total_elements:
        return
    
    # Calculate block boundaries
    block_start = pid * elements_per_block
    block_end = min((pid + 1) * elements_per_block, total_elements)
    
    # Process each element in the block
    for idx in range(block_start, block_end):
        # Calculate original indices: [batch, channels, h, w]
        batch_idx = idx // (channels * height * width)
        remaining = idx % (channels * height * width)
        channel_idx = remaining // (height * width)
        remaining = remaining % (height * width)
        h_idx = remaining // width
        w_idx = remaining % width
        
        # Calculate new indices after flatten and transpose: [batch, h*w, channels]
        new_idx = batch_idx * (channels * height * width) + \
                 (h_idx * width + w_idx) * channels + channel_idx
        
        # Load from input and store to output
        if idx < batch_size * channels * height * width:
            val = tl.load(input_ptr + idx)
            tl.store(output_ptr + new_idx, val)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    total_elements = batch_size * channels * height * width
    
    # Choose block size based on tensor dimensions for optimal performance
    if height * width >= 1000:  # Large spatial dimensions
        BLOCK_M = 32
        BLOCK_N = 32
    elif height * width >= 500:  # Medium spatial dimensions
        BLOCK_M = 64
        BLOCK_N = 16
    else:  # Small spatial dimensions
        BLOCK_M = 16
        BLOCK_N = 64
    
    elements_per_block = BLOCK_M * BLOCK_N
    num_programs = (total_elements + elements_per_block - 1) // elements_per_block
    
    output_tensor = torch.empty((batch_size, height * width, channels), 
                              dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    flatten_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output_tensor

def replacement_func():
    return optimized_flatten_transpose