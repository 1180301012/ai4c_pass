import torch
import triton
import triton.language as tl

# Simple pattern matching for flatten(2) followed by transpose(1, 2)
def pattern(conv_output):
    # matches: flatten(2) followed by transpose(1, 2)
    flattened = conv_output.flatten(2)  # flatten last two dimensions  
    transposed = flattened.transpose(1, 2)  # swap flattened dim with channel dim
    return transposed

def replacement_args(conv_output):
    return (conv_output,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for direct reshape from [B,C,H,W] to [B,H*W,C]"""
    pid = tl.program_id(0)
    
    # Calculate total elements per output tensor
    hw_elements = height * width
    total_elements = batch_size * channels * hw_elements
    elements_per_program = hw_elements
    
    # Calculate program range
    program_start = pid * elements_per_program
    element_offset = program_start + tl.arange(0, elements_per_program)
    mask = element_offset < total_elements
    
    if element_offset >= total_elements:
        return
    
    # Calculate 4D indices directly from linear offset
    # Efficient parallel computation without branching
    linear_idx = element_offset
    batch_idx = linear_idx // (channels * hw_elements)
    hw_idx = linear_idx % hw_elements
    channel_idx = (linear_idx // hw_elements) % channels
    
    # Direct memory access - no computation needed, just logical reorganization
    input_offset = batch_idx * (channels * height * width) + channel_idx * (height * width) + hw_idx
    output_offset = element_offset
    
    # Data movement with direct copy
    tl.store(output_ptr + output_offset, tl.load(input_ptr + input_offset, mask=mask, other=0.0), mask=mask)

@torch.fx.wrap
def optimized_flatten_transpose(conv_output):
    # Optimize flatten(2) + transpose(1, 2) using optimized Triton kernel
    # This avoids creating an intermediate flattened tensor and uses parallel GPU execution
    
    # Get the dimensions
    if conv_output.dim() != 4:
        # Fallback for non-4D tensors
        return conv_output.flatten(2).transpose(1, 2)
    
    batch_size = conv_output.shape[0]
    channels = conv_output.shape[1] 
    height = conv_output.shape[2]
    width = conv_output.shape[3]
    
    # Use optimized Triton kernel for large tensors
    # Only optimize if tensor is large enough to benefit from parallel processing
    total_elements = batch_size * channels * height * width
    
    # Small tensors use PyTorch reshape (faster for small data)
    if total_elements < 1024:  # Threshold for small tensors
        return conv_output.reshape(batch_size, height * width, channels)
    
    # Large tensors use optimized Triton kernel
    output = torch.empty(batch_size, height * width, channels, device=conv_output.device, dtype=conv_output.dtype)
    
    BLOCK_SIZE = 1024
    hw_elements = height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_reshape_kernel[(num_programs,)](
        input_ptr=conv_output,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_flatten_transpose