import torch
import triton
import triton.language as tl

def pattern(arg0, arg1, arg2, arg3, arg4, input_tensor):
    # This matches the tensor slicing pattern used in the computation
    # Note: The actual slice value may vary (128, 64, 2048, etc.) depending on the specific computation
    tmp_4 = input_tensor[slice(None, None, None), slice(0, None, None), slice(None, None, None), slice(None, None, None)]
    # The batch_norm result on arg4 (tmp_5) is also returned but not modified by this slice
    tmp_5 = arg4  # This is the batch_norm output, passed through unchanged
    return tmp_5, tmp_4

def replacement_args(arg0, arg1, arg2, arg3, arg4, input_tensor):
    return (arg0, arg1, arg2, arg3, arg4, input_tensor)

@triton.jit
def slice_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_channels,
    output_channels,
    height,
    width,
    slice_start: tl.constexpr,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Determine which channel block this program handles
    channel_idx = pid * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = (channel_idx + slice_start) < (slice_start + output_channels)
    
    # Process each spatial position in the channel block
    for h in range(height):
        for w in range(width):
            # Calculate linear index for this spatial position
            spatial_offset = h * width + w
            
            # Load input data for this batch and spatial position
            input_offset = spatial_offset * input_channels + channel_idx + slice_start
            input_data = tl.load(input_ptr + input_offset, mask=channel_mask, other=0.0)
            
            # Store output data (channel indexing is automatic)
            output_offset = spatial_offset * output_channels + channel_idx
            tl.store(output_ptr + output_offset, input_data, mask=channel_mask)

@torch.fx.wrap
def optimized_slice(input_tensor, slice_start=0):
    # Check input tensor shape: [batch_size, input_channels, height, width]
    batch_size, input_channels, height, width = input_tensor.shape
    
    # Calculate output channels (from slice_start to end)
    output_channels = input_channels - slice_start
    
    # Create output tensor
    output = torch.empty((batch_size, output_channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size for channels - should divide output_channels evenly
    BLOCK_SIZE_CHANNELS = min(64, triton.next_power_of_2(output_channels))
    
    # Number of programs needed to cover all channels
    n_programs = (output_channels + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Launch kernel
    slice_kernel[(n_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        height=height,
        width=width,
        slice_start=slice_start,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
    )
    
    return output

def replacement_func():
    return optimized_slice