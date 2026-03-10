import torch
import triton
import triton.language as tl

@triton.jit
def channel_concat_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    batch_size,
    channels1,
    channels2,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    total_elements = batch_size * (channels1 + channels2) * height * width
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    if pid >= num_programs:
        return
    
    # Calculate indices
    output_idx = offsets
    batch_idx = output_idx // ((channels1 + channels2) * height * width)
    remaining = output_idx % ((channels1 + channels2) * height * width)
    channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    height_idx = spatial_idx // width
    width_idx = spatial_idx % width
    
    # Check if this comes from first or second tensor
    if channel_idx < channels1:
        # From first tensor
        input_offset = batch_idx * channels1 * height * width + \
                      channel_idx * height * width + spatial_idx
        result = tl.load(x1_ptr + input_offset, mask=mask, other=0.0)
    else:
        # From second tensor
        input_offset = batch_idx * channels2 * height * width + \
                      (channel_idx - channels1) * height * width + spatial_idx
        result = tl.load(x2_ptr + input_offset, mask=mask, other=0.0)
    
    # Store result
    output_offset = output_idx
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def optimized_channel_concat(x1, x2, dim=1):
    # Input tensors should have same shape except for concatenation dimension
    assert x1.shape[0] == x2.shape[0], f"Batch size mismatch: {x1.shape[0]} vs {x2.shape[0]}"
    assert dim == 1, f"Only implemented for dim=1, got {dim}"
    
    batch_size, channels1, height, width = x1.shape
    channels2 = x2.shape[1]
    
    # Create output tensor
    output = torch.empty((batch_size, channels1 + channels2, height, width), 
                        dtype=x1.dtype, device=x1.device)
    
    # Flatten inputs for easier kernel processing
    x1_flat = x1.view(batch_size * channels1 * height * width)
    x2_flat = x2.view(batch_size * channels2 * height * width)
    output_flat = output.view(batch_size * (channels1 + channels2) * height * width)
    
    # Calculate kernel launch parameters
    total_elements = batch_size * (channels1 + channels2) * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    channel_concat_kernel[(num_programs,)](
        x1_flat,
        x2_flat,
        output_flat,
        batch_size,
        channels1,
        channels2,
        height,
        width,
        BLOCK_SIZE
    )
    
    return output

# Pattern matching for concatenation along channel dimension
def pattern(x1, x2):
    return torch.cat([x1, x2], 1)

# Argument extraction
def replacement_args(x1, x2):
    return (x1, x2)

# Replacement function
def replacement_func():
    return optimized_channel_concat