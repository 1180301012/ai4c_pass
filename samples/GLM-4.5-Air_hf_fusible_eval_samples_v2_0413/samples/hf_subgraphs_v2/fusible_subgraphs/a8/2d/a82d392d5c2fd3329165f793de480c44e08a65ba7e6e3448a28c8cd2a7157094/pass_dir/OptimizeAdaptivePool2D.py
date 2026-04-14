import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Match adaptive_avg_pool2d operation
    return torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate total number of output elements
    output_n_elements = batch_size * out_channels * out_height * out_width
    element_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = element_offsets < output_n_elements
    
    if not tl.any(mask):
        return
    
    # Convert element offset to batch, channel, height, width indices
    flat_idx = element_offsets
    batch_idx = flat_idx // (out_channels * out_height * out_width)
    remainder = flat_idx % (out_channels * out_height * out_width)
    channel_idx = remainder // (out_height * out_width)
    spatial_idx = remainder % (out_height * out_width)
    height_idx = spatial_idx // out_width
    width_idx = spatial_idx % out_width
    
    # Handle pooling only for channels 0 to min(in_channels, out_channels)
    if channel_idx < min(in_channels, out_channels):
        # Calculate the size of each pool window
        # Since adaptive_avg_pool2d with (32, 24) on (64, 48) input
        # uses 2x2 pooling windows
        pool_h = in_height // out_height
        pool_w = in_width // out_width
        
        # Calculate the starting positions of the pool window
        pool_start_h = height_idx * pool_h
        pool_start_w = width_idx * pool_w
        
        # Sum all values in the pool window
        pool_sum = 0.0
        pool_count = 0
        
        # Iterate over the pool window
        for dy in range(pool_h):
            for dx in range(pool_w):
                h = pool_start_h + dy
                w = pool_start_w + dx
                
                # Ensure we're within bounds for the last window
                if h < in_height and w < in_width:
                    input_offset = batch_idx * in_channels * in_height * in_width + \
                                 channel_idx * in_height * in_width + h * in_width + w
                    
                    # Load input value
                    val = tl.load(input_ptr + input_offset)
                    pool_sum += val
                    pool_count += 1
        
        # Calculate average
        if pool_count > 0:
            avg_val = pool_sum / pool_count
            output_offset = batch_idx * out_channels * out_height * out_width + \
                          channel_idx * out_height * out_width + height_idx * out_width + width_idx
            tl.store(output_ptr + output_offset, avg_val, mask=mask)
    
    # For channels beyond input_channels, fill with zeros (this shouldn't happen)
    # or pad the input if needed for output_channels > input_channels
    elif in_channels < out_channels:
        output_offset = batch_idx * out_channels * out_height * out_width + \
                      channel_idx * out_height * out_width + height_idx * out_width + width_idx
        tl.store(output_ptr + output_offset, 0.0, mask=mask)

@triton.jit
def optimized_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that processes elements in the block"""
    # Program ID for 1D grid
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process each element in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * in_channels * out_height * out_width)
    
    # Kernel handles mask naturally
    
    # Each program processes one element: use program_id directly
    idx = pid
    
    # Check if idx is within bounds
    if idx < (batch_size * in_channels * out_height * out_width):
        # Convert linear index to coordinates
        batch_idx = idx // (in_channels * out_height * out_width)
        remainder = idx % (in_channels * out_height * out_width)
        channel_idx = remainder // (out_height * out_width)
        remainder = remainder % (out_height * out_width)
        out_y = remainder // out_width
        out_x = remainder % out_width
        
        # Calculate pooling window size
        pool_h = in_height // out_height if out_height > 0 else 1
        pool_w = in_width // out_width if out_width > 0 else 1
        pool_h = max(1, pool_h)
        pool_w = max(1, pool_w)
        
        # Calculate input window start position
        in_y_start = out_y * pool_h
        in_x_start = out_x * pool_w
        
        # Sum pooling window
        pool_sum = 0.0
        pool_count = 0
        
        for dy in range(pool_h):
            for dx in range(pool_w):
                in_y = in_y_start + dy
                in_x = in_x_start + dx
                
                if in_y < in_height and in_x < in_width:
                    in_offset = batch_idx * in_channels * in_height * in_width + \
                               channel_idx * in_height * in_width + in_y * in_width + in_x
                    val = tl.load(input_ptr + in_offset)
                    pool_sum += val
                    pool_count += 1
        
        # Store averaged result
        if pool_count > 0:
            avg_val = pool_sum / tl.cast(pool_count, tl.float32)
            out_offset = batch_idx * in_channels * out_height * out_width + \
                       channel_idx * out_height * out_width + out_y * out_width + out_x
            tl.store(output_ptr + out_offset, avg_val)

@torch.fx.wrap
def optimized_adaptive_avg_pool2d(in_0):
    input_shape = in_0.shape
    batch_size = input_shape[0]
    in_channels = input_shape[1]
    in_height = input_shape[2]
    in_width = input_shape[3]
    
    output_height = 32
    output_width = 24
    out_channels = in_channels  # Adaptive pooling preserves channel count
    
    # Create output tensor
    output_shape = (batch_size, out_channels, output_height, output_width)
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use the simple kernel - each program processes one element
    output_n_elements = batch_size * in_channels * output_height * output_width
    grid = (output_n_elements,)  # One program per output element
    
    optimized_adaptive_avg_pool2d_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=output_height,
        out_width=output_width,
        BLOCK_SIZE=1,  # Block size of 1 since each program does one element
    )
    
    return output

def replacement_func():
    return optimized_adaptive_avg_pool2d