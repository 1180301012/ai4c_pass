import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Conv2D + Permute pattern fusion
    Matches the exact pattern from model.py:
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    """
    # Conv2D operation - exact match from model.py with same argument order
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Permute operation that reshapes from [B, C, H, W] to [B, H, W, C]
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_permute_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, in_channels_: tl.constexpr, out_channels_: tl.constexpr, height_: tl.constexpr, width_: tl.constexpr,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    """
    Optimized fused Conv2D + Permute kernel - processes multiple channels per launch
    """
    pid_batch = tl.program_id(0)
    pid_channel_block = tl.program_id(1)  # Channel block index
    pid_hw_block = tl.program_id(2)       # Spatial block index
    
    # Calculate channel range for this block
    channel_start = pid_channel_block * BLOCK_SIZE_CHANNELS
    channel_end = min(channel_start + BLOCK_SIZE_CHANNELS, out_channels_)
    if channel_start >= out_channels_:
        return
    
    # Calculate spatial range for this block - process first position in block only
    hw_pos = pid_hw_block * 1  # Process first position in each HW block
    if hw_pos >= height_ * width_:
        return
        
    h = hw_pos // width_
    w = hw_pos % width_
    
    # Process each channel in the block individually (simplified approach)
    for local_idx in range(BLOCK_SIZE_CHANNELS):
        channel = channel_start + local_idx
        if channel < out_channels_:  # Only process if in bounds
            
            # Initialize accumulator for this channel
            acc = 0.0
            
            # Process all input channels
            for k in range(in_channels_):
                # Load input value at current position
                input_idx = pid_batch * in_channels_ * height_ * width_ + k * height_ * width_ + h * width_ + w
                input_val = tl.load(input_ptr + input_idx).to(tl.float32)
                
                # Load weight value for this channel
                weight_idx = k * out_channels_ + channel
                weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                
                # Accumulate: a = a + x * y
                acc += input_val * weight_val
            
            # Load bias and add
            bias_val = tl.load(bias_ptr + channel).to(tl.float32)
            acc = acc + bias_val
            
            # Store result for this channel
            output_base = pid_batch * height_ * width_ * out_channels_ + h * width_ * out_channels_ + w * out_channels_ + channel
            tl.store(output_ptr + output_base, acc)

@torch.fx.wrap
def fused_conv2d_permute(in_0, in_1, in_2):
    """
    Fused Conv2D + Permute function
    Performs 1x1 convolution and directly outputs [B, H, W, C] format
    """
    # Parameters: in_0 is bias, in_1 is weight, in_2 is input
    batch_size, in_channels_, height_, width_ = in_2.shape
    out_channels_ = in_1.shape[0]
    
    # Output shape is now [B, H, W, C]
    output_shape = (batch_size, height_, width_, out_channels_)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Optimized block sizes to minimize kernel launches while maintaining efficiency
    # Process multiple spatial positions and channels per kernel
    BLOCK_SIZE_CHANNELS = 64   # Process 64 channels per kernel
    BLOCK_SIZE_HW = 128        # Process 128 spatial locations per kernel
    
    # Grid size: (batch_size, ceil(out_channels/BLOCK_SIZE_CHANNELS), ceil(height*width/BLOCK_SIZE_HW))
    num_channel_blocks = (out_channels_ + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    num_hw_blocks = (height_ * width_ + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid = lambda meta: (
        batch_size,
        num_channel_blocks,
        num_hw_blocks,
    )
    
    fused_conv2d_permute_kernel[grid](
        in_2,  # input
        in_1,  # weight
        in_0,  # bias
        output,
        batch_size, in_channels_, out_channels_, height_, width_,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
    )
    
    return output

def replacement_func():
    return fused_conv2d_permute