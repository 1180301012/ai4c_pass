import torch
import triton
import triton.language as tl

@triton.jit
def fused_pool_flatten_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * channels  # After pooling and flattening
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # For adaptive_avg_pool2d to (1,1), each output element is the average of the entire spatial channel
    # Load complete spatial dimension for each channel
    input_per_channel = input_ptr + tl.arange(0, channels) * height * width
    
    # Compute average for each channel (reduce over spatial dimensions)
    for h in range(height):
        for w in range(width):
            channel_idx = tl.arange(0, channels)
            spatial_offset = h * width + w
            input_offset = input_per_channel + spatial_offset
            
            # Load spatial data for all channels
            spatial_data = tl.load(input_offset, mask=True, other=0.0)
            
            if h == 0 and w == 0:
                # Initialize accumulator
                channel_sums = spatial_data
            else:
                # Accumulate spatial data
                channel_sums += spatial_data
    
    # Compute averages (sum / spatial_size)
    spatial_size = height * width
    channel_averages = channel_sums / float(spatial_size)
    
    # Store result (flattened)
    tl.store(output_ptr + offset, channel_averages, mask=mask)

def pattern(tmp_5):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    # Dropout with p=0.0 does nothing, so we eliminate it
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8

def replacement_args(tmp_5):
    return (tmp_5,)

@torch.fx.wrap
def fused_pool_flatten_forward(tmp_5):
    """Fused adaptive avg pool2d + flatten with dropout elimination"""
    batch_size, channels, height, width = tmp_5.shape
    
    # Calculate optimal block size and grid
    total_elements = batch_size * channels  # After pooling and flattening
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (flattened)
    output = torch.empty((batch_size, channels), dtype=tmp_5.dtype, device=tmp_5.device)
    
    # Launch fused kernel
    fused_pool_flatten_kernel[(num_programs,), (1,)](
        tmp_5,
        output,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_pool_flatten_forward