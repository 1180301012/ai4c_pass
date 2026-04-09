import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    """Pattern: adaptive_avg_pool2d + flatten fusion"""
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized global average pooling kernel"""
    pid = tl.program_id(0)
    
    # Each program handles one batch element
    batch_offset = pid * channels
    
    # Compute global average for each channel in this batch element
    for channel_idx in tl.range(0, channels, BLOCK_SIZE):
        # Create mask for this channel block
        channel_mask = (channel_idx + tl.arange(0, BLOCK_SIZE)) < channels
        
        # Initialize sum for this channel block
        channel_sum = tl.zeros(1, dtype=tl.float32)
        element_count = tl.zeros(1, dtype=tl.int32)
        
        # Block processing for better memory access
        for h_idx in range(height):
            for w_idx in range(width):
                # Compute linear index for this pixel position across all channels in block
                pixel_base = batch_offset + (h_idx * width + w_idx) * channels
                
                # Load channel block at this pixel position
                channel_pixels = tl.load(
                    input_ptr + pixel_base + channel_idx,
                    mask=channel_mask,
                    other=0.0
                )
                
                # Accumulate sum
                tl.store(channel_sum, channel_sum + tl.sum(channel_pixels))
                tl.store(element_count, element_count + tl.sum(channel_mask))
        
        # Compute average and store
        avg_value = channel_sum / element_count if element_count > 0 else 0.0
        
        # Store result for this channel block
        output_indices = channel_idx + tl.arange(0, BLOCK_SIZE)
        tl.store(
            output_ptr + pid * channels + output_indices,
            avg_value,
            mask=channel_mask
        )

@torch.fx.wrap
def fused_global_avg_pool(input_tensor):
    """Fused global average pooling + flatten operation"""
    batch_size, channels, height, width = input_tensor.shape
    
    # Determine optimal block size
    total_elements = batch_size * channels
    BLOCK_SIZE = 128  # Can be tuned for different GPU architectures
    
    # Calculate grid dimensions
    num_batches = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_batches, 1, 1)
    
    # Create output tensor
    output_shape = (batch_size, channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    global_avg_pool_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_global_avg_pool