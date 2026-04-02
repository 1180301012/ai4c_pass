import torch
import triton
import triton.language as tl

@triton.jit
def optimized_concatenate_channels_kernel(
    input1_ptr,
    input2_ptr, 
    output_ptr,
    batch_size,
    channels1,
    channels2,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for channel concatenation
    Each program handles one batch element and one spatial location
    """
    pid = tl.program_id(0)
    
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    
    if batch_idx >= batch_size:
        return
    
    y = spatial_idx // width
    x = spatial_idx % width
    
    total_channels = channels1 + channels2
    
    # Process channels in chunks
    for c_start in range(0, total_channels, BLOCK_SIZE):
        c_end = min(c_start + BLOCK_SIZE, total_channels)
        channel_block = tl.arange(c_start, c_end)
        mask = channel_block < total_channels
        
        # Calculate output offset
        output_offset = (batch_idx * total_channels * height * width + 
                        y * total_channels * width + x * total_channels + channel_block)
        
        # Load first input block (channels 0 to channels1-1)
        if c_start < channels1:
            load_channels = channel_block[channel_block < channels1]
            load_mask = load_channels < channels1
            offset1 = (batch_idx * channels1 * height * width + 
                       y * channels1 * width + x * channels1 + load_channels)
            vals1 = tl.load(input1_ptr + offset1, mask=load_mask)
            tl.store(output_ptr + output_offset + load_channels - c_start, vals1, mask=load_mask)
        
        # Load second input block (channels channels1 to channels1+channels2-1)
        if c_start < channels1 + channels2:
            second_start = max(0, c_start - channels1)
            second_channels = channel_block[channel_block >= channels1]
            second_mask = (second_channels >= channels1) & (second_channels < channels1 + channels2)
            if tl.any(second_mask):
                offset2 = (batch_idx * channels2 * height * width + 
                           y * channels2 * width + x * channels2 + second_channels - channels1)
                vals2 = tl.load(input2_ptr + offset2, mask=second_mask)
                tl.store(output_ptr + output_offset + (second_channels - c_start)[second_mask], 
                        vals2, mask=second_mask)

@torch.fx.wrap 
def optimized_concatenate_channels(input1, input2, dim=1):
    """
    Optimized channel concatenation using Triton
    """
    # Validate inputs
    if input1.shape[0] != input2.shape[0] or input1.shape[2] != input2.shape[2] or input1.shape[3] != input2.shape[3]:
        raise ValueError("Input tensors must be compatible for concatenation")
    
    # Ensure tensors are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    
    batch_size, channels1, height, width = input1.shape
    channels2 = input2.shape[1]
    
    # Create output tensor
    output_shape = (batch_size, channels1 + channels2, height, width)
    output = torch.empty(output_shape, dtype=input1.dtype, device=input1.device)
    
    # Set up grid and launch kernel
    grid = (batch_size * height * width,)
    BLOCK_SIZE = 128  # Optimal for channel processing
    
    optimized_concatenate_channels_kernel[grid](
        input1_ptr=input1,
        input2_ptr=input2,
        output_ptr=output,
        batch_size=batch_size,
        channels1=channels1,
        channels2=channels2,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Pattern matching function
def pattern(tensor1, tensor2, dim=None):
    """
    Match torch.cat operation along channel dimension
    """
    if dim is None:
        dim = 1  # Default to channel dimension
    result = torch.cat([tensor1, tensor2], dim)
    return result

# Argument extraction function
def replacement_args(tensor1, tensor2, dim=None):
    return (tensor1, tensor2, dim)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_concatenate_channels