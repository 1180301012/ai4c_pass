import torch
import triton
import triton.language as tl

def pattern(relu_output):
    # Match the global average pooling + flatten operation sequence
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(relu_output, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_6, tmp_7

def replacement_args(relu_output):
    return (relu_output,)

@triton.jit
def fused_global_pool_flatten_kernel(
    relu_output_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    # Each program handles one channel in the output
    pid = tl.program_id(0)
    
    # Calculate the output for this channel (global average pooled)
    channel_start = pid * BLOCK_SIZE_CHANNELS
    channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_offsets < channels
    
    # Initialize sum for this channel
    channel_sum = 0.0
    
    # Loop over spatial dimensions to compute global average
    for h in range(height):
        for w in range(width):
            # Calculate offset for spatial position (h, w)
            spatial_offset = (h * width + w) * batch_size * channels + channel_offsets
            
            # Load data for all channels at this spatial position
            spatial_values = tl.load(relu_output_ptr + spatial_offset, mask=channel_offsets < channels, other=0.0)
            
            # Accumulate sum
            channel_sum += tl.sum(spatial_values * channel_mask)
    
    # Compute average: sum / (height * width * batch_size)
    pool_size = height * width * batch_size
    channel_avg = channel_sum / pool_size
    
    # Store the result in the flattened output
    out_ptr_offset = pid
    tl.store(out_ptr + out_ptr_offset, channel_avg, pid < channels)

@torch.fx.wrap
def fused_global_pool_flatten_operation(relu_output):
    batch_size, channels, height, width = relu_output.shape
    
    # Set up Triton kernel launch parameters
    BLOCK_SIZE_CHANNELS = 2048  # Process multiple channels per thread
    num_programs = (channels + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Create output tensor (flattened result: [batch_size, channels])
    out = torch.empty((batch_size, channels), dtype=relu_output.dtype, device=relu_output.device)
    
    # Launch Triton kernel
    fused_global_pool_flatten_kernel[(num_programs,)](
        relu_output_ptr=relu_output,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
    )
    
    # Return both the pooled and flattened results to match the pattern
    pooled_out = out.view(batch_size, channels, 1, 1)
    return pooled_out, out

def replacement_func():
    return fused_global_pool_flatten_operation