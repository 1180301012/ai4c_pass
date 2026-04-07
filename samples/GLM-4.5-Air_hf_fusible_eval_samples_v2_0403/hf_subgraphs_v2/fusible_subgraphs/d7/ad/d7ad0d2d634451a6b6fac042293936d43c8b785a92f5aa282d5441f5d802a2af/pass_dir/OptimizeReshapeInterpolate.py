import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: Reshape + Interpolate optimization"""
    # Reshape from [batch_size, 768, seq_len] to [batch_size, channels, height, width]
    # where height=16, width=16, and channels is inferred automatically
    reshaped = x.reshape(x.shape[0], -1, 16, 16)
    result = torch.nn.functional.interpolate(reshaped, size=(128, 128), mode='bilinear', align_corners=False)
    return result

def replacement_args(x):
    """Extract arguments for the optimized reshape+interpolate"""
    return (x,)

@triton.jit
def interpolate_kernel(
    x_ptr, out_ptr,
    batch_size: tl.constexpr, channels: tl.constexpr, 
    in_height: tl.constexpr, in_width: tl.constexpr,
    out_height: tl.constexpr, out_width: tl.constexpr,
    BLOCK_BATCH: tl.constexpr, BLOCK_CHANNEL: tl.constexpr
):
    """Optimized bilinear interpolation kernel"""
    # Calculate grid positions
    b = tl.program_id(0)
    c = tl.program_id(1)
    local_id = tl.program_id(2)
    
    # Each thread processes a block of output pixels
    out_y_base = (local_id // 8) * 8
    out_x_base = (local_id % 8) * 8
    
    # Scale factors
    scale_y = (in_height - 1) / max(out_height - 1, 1)
    scale_x = (in_width - 1) / max(out_width - 1, 1)
    
    # Process 8x8 output block per thread
    for dy in range(8):
        for dx in range(8):
            out_y = out_y_base + dy
            out_x = out_x_base + dx
            
            if out_y >= out_height or out_x >= out_width:
                continue
                
            # Calculate corresponding input coordinates
            in_y = out_y * scale_y
            in_x = out_x * scale_x
            
            # Get integer parts and fractional parts
            y0 = tl.math.floor(in_y)
            x0 = tl.math.floor(in_x)
            y1 = min(y0 + 1, in_height - 1)
            x1 = min(x0 + 1, in_width - 1)
            
            # Calculate interpolation weights
            wy = in_y - y0
            wx = in_x - x0
            
            # Load four corner values
            base_offset = b * channels * in_height * in_width + c * in_height * in_width
            offset_00 = base_offset + y0 * in_width + x0
            offset_01 = base_offset + y0 * in_width + x1
            offset_10 = base_offset + y1 * in_width + x0
            offset_11 = base_offset + y1 * in_width + x1
            
            # Load with bounds checking
            v00 = tl.load(x_ptr + offset_00, mask=(y0 < in_height) & (x0 < in_width))
            v01 = tl.load(x_ptr + offset_01, mask=(y0 < in_height) & (x1 < in_width))
            v10 = tl.load(x_ptr + offset_10, mask=(y1 < in_height) & (x0 < in_width))
            v11 = tl.load(x_ptr + offset_11, mask=(y1 < in_height) & (x1 < in_width))
            
            # Bilinear interpolation
            v0 = v00 * (1 - wx) + v01 * wx
            v1 = v10 * (1 - wx) + v11 * wx
            result = v0 * (1 - wy) + v1 * wy
            
            # Store result
            out_offset = b * channels * out_height * out_width + c * out_height * out_width + out_y * out_width + out_x
            tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def optimized_interpolate(x):
    """Optimized reshape + interpolate with fused kernel"""
    batch_size, hidden_dim, seq_len = x.shape
    
    # Calculate reshape dimensions
    seq_height = 16
    seq_width = 16
    channels = hidden_dim * seq_len // (seq_height * seq_width)
    
    # Reshape input to [batch_size, channels, height, width]
    x_reshaped = x.reshape(batch_size, channels, seq_height, seq_width)
    
    # Create output tensor
    out_height, out_width = 128, 128
    output = torch.empty((batch_size, channels, out_height, out_width), 
                        dtype=x.dtype, device=x.device)
    
    # Choose block sizes
    BLOCK_BATCH = 1
    BLOCK_CHANNEL = 8
    
    # Calculate grid dimensions
    grid_batch = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_channel = (channels + BLOCK_CHANNEL - 1) // BLOCK_CHANNEL
    grid_local = ((out_height + 7) // 8) * ((out_width + 7) // 8)
    
    # Launch kernel for each batch/channel combination
    for b in range(0, batch_size, BLOCK_BATCH):
        for c in range(0, channels, BLOCK_CHANNEL):
            actual_batch = min(BLOCK_BATCH, batch_size - b)
            actual_channel = min(BLOCK_CHANNEL, channels - c)
            
            interpolate_kernel[(actual_batch, actual_channel, grid_local)](
                x_ptr=x_reshaped[b:b+actual_batch, c:c+actual_channel].contiguous().data_ptr(),
                out_ptr=output[b:b+actual_batch, c:c+actual_channel].data_ptr(),
                batch_size=actual_batch, channels=actual_channel,
                in_height=seq_height, in_width=seq_width,
                out_height=out_height, out_width=out_width,
                BLOCK_BATCH=BLOCK_BATCH, BLOCK_CHANNEL=BLOCK_CHANNEL
            )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_interpolate