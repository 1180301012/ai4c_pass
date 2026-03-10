import torch
import triton
import triton.language as tl

# Pattern matching for flatten(2) followed by transpose(1, 2)
def pattern(conv_output):
    # matches: tmp_7 = tmp_6.flatten(2) followed by tmp_8 = tmp_7.transpose(1, 2)
    flattened = conv_output.flatten(2)  # flatten last two dimensions
    transposed = flattened.transpose(1, 2)  # swap flattened dim with channel dim
    return transposed

def replacement_args(conv_output):
    return (conv_output,)

@triton.jit
def fuse_flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total elements per program
    total_hw_elements = height * width
    total_elements = batch_size * channels * total_hw_elements
    elements_per_program = total_hw_elements
    
    # Calculate the starting position for this program
    program_base = pid * elements_per_program
    element_offset = program_base + tl.arange(0, elements_per_program)
    mask = element_offset < total_elements
    
    # Calculate indices in the original 4D tensor [B, C, H, W]
    linear_idx = element_offset
    # Batch and channel indices
    batch_idx = linear_idx // (channels * total_hw_elements)
    hw_idx = linear_idx % total_hw_elements
    channel_idx = (linear_idx // total_hw_elements) % channels
    
    # Load from conv2d output [B, C, H, W]
    input_offset = batch_idx * (channels * height * width) + channel_idx * (height * width) + hw_idx
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store directly to output [B, H*W, C] - same memory layout, just logical reshape
    output_offset = element_offset
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap  
def optimized_flatten_transpose(conv_output):
    # Only optimize if input is 4D (conv2d output)
    if conv_output.dim() != 4:
        # Fallback to original behavior
        return conv_output.flatten(2).transpose(1, 2)
    
    B, C, H, W = conv_output.shape
    total_elements = conv_output.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(conv_output)  # Same tensor, just reshaped logically
    
    fuse_flatten_transpose_kernel[(num_programs,)](
        input_ptr=conv_output,
        output_ptr=output,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Logical reshape: just change the view to [B, H*W, C]
    return output.reshape(B, H * W, C)

def replacement_func():
    return optimized_flatten_transpose