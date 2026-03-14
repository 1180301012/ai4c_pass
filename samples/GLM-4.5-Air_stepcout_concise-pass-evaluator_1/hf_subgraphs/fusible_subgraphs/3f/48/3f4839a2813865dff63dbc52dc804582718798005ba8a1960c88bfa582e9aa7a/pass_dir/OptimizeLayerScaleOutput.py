import torch
import triton
import triton.language as tl

def pattern(layer_scale):
    """
    Pattern: Layer scale output computation with sequential unsqueeze operations
    Original: 
        tmp_8 = layer_scale.unsqueeze(-1)  # [48] → [48, 1]
        tmp_9 = tmp_8.unsqueeze(-1)         # [48, 1] → [48, 1, 1]
        result = tmp_9
    """
    tmp_8 = layer_scale.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_9

def replacement_args(layer_scale):
    return (layer_scale,)

@triton.jit
def layer_scale_expand_kernel(
    layer_scale_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load layer scale values for each position
    channel_idx = offsets // (height * width)  # Group by channel
    layer_scale_val = tl.load(layer_scale_ptr + channel_idx % channels, mask=channel_idx < channels)
    
    # Broadcast layer scale value to all spatial positions
    output = layer_scale_val
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_scale_expand(layer_scale):
    # Expand layer_scale from [C] to [1, C, 1, 1] efficiently
    expanded = layer_scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    batch_size, channels, height, width = expanded.shape
    total_elements = batch_size * channels * height * width
    
    # For very large tensors, use Triton kernel for optimal performance
    if total_elements > 100000:  # Use Triton for large tensors
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty_like(expanded)
        
        layer_scale_expand_kernel[(num_programs,)](
            layer_scale_ptr=layer_scale,
            output_ptr=output,
            batch_size=1,
            channels=channels,
            height=1,
            width=1,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output
    else:
        # For small tensors, simple expansion is fine
        return expanded

def replacement_func():
    return optimized_layer_scale_expand