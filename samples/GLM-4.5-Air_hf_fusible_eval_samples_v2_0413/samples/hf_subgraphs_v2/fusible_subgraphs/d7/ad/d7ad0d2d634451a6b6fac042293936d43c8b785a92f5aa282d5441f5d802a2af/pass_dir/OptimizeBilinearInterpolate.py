import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    # Bilinear interpolation operation
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp_5

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def bilinear_interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Each program handles a spatial region of the output
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_y = tl.program_id(2)
    pid_x = tl.program_id(3)
    
    # Calculate the range of this block
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    channel_offsets = pid_channel * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    y_offsets = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    x_offsets = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    
    # Create masks
    batch_mask = batch_offsets < batch_size
    channel_mask = channel_offsets < channels
    y_mask = y_offsets < out_height
    x_mask = x_offsets < out_width
    
    # Calculate scale factors
    y_scale = (in_height - 1) / (out_height - 1) if out_height > 1 else 0
    x_scale = (in_width - 1) / (out_width - 1) if out_width > 1 else 0
    
    # Process each position in the output block
    for i in range(BLOCK_BATCH):
        for j in range(BLOCK_CHANNELS):
            for k in range(BLOCK_Y):
                for l in range(BLOCK_X):
                    batch_idx = batch_offsets[i]
                    channel_idx = channel_offsets[j]
                    out_y = y_offsets[k]
                    out_x = x_offsets[l]
                    
                    if (batch_idx < batch_size and channel_idx < channels and 
                        out_y < out_height and out_x < out_width):
                        
                        # Map output coordinates to input coordinates
                        in_y = out_y * y_scale
                        in_x = out_x * x_scale
                        
                        # Get integer and fractional parts
                        y0 = tl.int_floor(in_y)
                        x0 = tl.int_floor(in_x)
                        y1 = y0 + 1
                        x1 = x0 + 1
                        
                        # Get fractional components
                        dy = in_y - y0
                        dx = in_x - x0
                        
                        # Clamp coordinates to input bounds
                        y0 = max(0, min(y0, in_height - 1))
                        x0 = max(0, min(x0, in_width - 1))
                        y1 = max(0, min(y1, in_height - 1))
                        x1 = max(0, min(x1, in_width - 1))
                        
                        # Calculate bilinear interpolation weights
                        w00 = (1 - dy) * (1 - dx)
                        w01 = (1 - dy) * dx
                        w10 = dy * (1 - dx)
                        w11 = dy * dx
                        
                        # Load four neighboring pixels
                        # Top-left (y0, x0)
                        idx_00 = (batch_idx * channels + channel_idx) * in_height * in_width + y0 * in_width + x0
                        val_00 = tl.load(input_ptr + idx_00, other=0.0)
                        
                        # Top-right (y0, x1)
                        idx_01 = (batch_idx * channels + channel_idx) * in_height * in_width + y0 * in_width + x1
                        val_01 = tl.load(input_ptr + idx_01, other=0.0)
                        
                        # Bottom-left (y1, x0)
                        idx_10 = (batch_idx * channels + channel_idx) * in_height * in_width + y1 * in_width + x0
                        val_10 = tl.load(input_ptr + idx_10, other=0.0)
                        
                        # Bottom-right (y1, x1)
                        idx_11 = (batch_idx * channels + channel_idx) * in_height * in_width + y1 * in_width + x1
                        val_11 = tl.load(input_ptr + idx_11, other=0.0)
                        
                        # Compute bilinear interpolation
                        result = (w00 * val_00 + w01 * val_01 + w10 * val_10 + w11 * val_11)
                        
                        # Store output
                        out_idx = (batch_idx * channels + channel_idx) * out_height * out_width + out_y * out_width + out_x
                        tl.store(output_ptr + out_idx, result)

@torch.fx.wrap
def optimized_bilinear_interpolate(input_tensor):
    batch_size = input_tensor.shape[0]
    channels = input_tensor.shape[1]
    in_height = input_tensor.shape[2]
    in_width = input_tensor.shape[3]
    out_height = 128
    out_width = 128
    
    # Output shape
    output_shape = (batch_size, channels, out_height, out_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions with tile-based approach
    BLOCK_BATCH = 1   # Process 1 batch element per program
    BLOCK_CHANNELS = 32  # Process 32 channels per program
    BLOCK_Y = 16     # Process 16 height positions per program
    BLOCK_X = 16     # Process 16 width positions per program
    
    grid_batch = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_channels = (channels + BLOCK_CHANNELS - 1) // BLOCK_CHANNELS
    grid_y = (out_height + BLOCK_Y - 1) // BLOCK_Y
    grid_x = (out_width + BLOCK_X - 1) // BLOCK_X
    
    grid = (grid_batch, grid_channels, grid_y, grid_x)
    
    bilinear_interpolate_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_CHANNELS=BLOCK_CHANNELS,
        BLOCK_Y=BLOCK_Y,
        BLOCK_X=BLOCK_X
    )
    
    return output

def replacement_func():
    return optimized_bilinear_interpolate