import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_maxpool_concat_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs and offsets
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Base offset for this batch and channel
    base_offset = batch_id * channels * height * width
    
    # Process spatial elements
    for h in range(0, height, BLOCK_SIZE):
        for w in range(0, width, BLOCK_SIZE):
            # Calculate tile bounds
            h_end = min(h + BLOCK_SIZE, height)
            w_end = min(w + BLOCK_SIZE, width)
            
            # Load input tile
            offset = base_offset + channel_id * height * width + h * width + w
            inputs = tl.load(x_ptr + offset, mask=(h < height) & (w < width), other=0.0)
            
            # Apply ReLU
            relu_out = tl.max(inputs, 0)
            
            # Store ReLU result (tmp_0)
            out_offset = base_offset + channel_id * height * width + h * width + w
            tl.store(out_ptr + out_offset, relu_out, mask=(h < height) & (w < width))
            
            # Calculate max_pool2d for 3 different offsets
            # Each max_pool2d operates on the same ReLU result but with different window positions
            
            # Max pool 1 (tmp_1) - offset calculation for pooling
            for pool_idx in range(3):
                pool_out_channels = channels + pool_idx * channels
                h_pool_start = max(0, h - 2)
                w_pool_start = max(0, w - 2)
                
                max_val = -float('inf')
                valid_values = 0
                
                # 5x5 max pool window
                for dh in range(5):
                    for dw in range(5):
                        h_current = h_pool_start + dh
                        w_current = w_pool_start + dw
                        
                        if 0 <= h_current < height and 0 <= w_current < width:
                            input_offset = base_offset + channel_id * height * width + h_current * width + w_current
                            input_val = tl.load(x_ptr + input_offset)
                            input_val = tl.max(input_val, 0)  # ReLU
                            max_val = tl.max(max_val, input_val)
                            valid_values += 1
                
                if valid_values > 0:
                    store_offset = base_offset + pool_out_channels * height * width + h * width + w
                    tl.store(out_ptr + store_offset, max_val, mask=(h < height) & (w < width))

@torch.fx.wrap
def fused_relu_maxpool_concat(x):
    batch_size, channels, height, width = x.shape
    
    # Output will have 4x channels (original + 3 pooled)
    out_channels = channels * 4
    out = torch.empty((batch_size, out_channels, height, width), 
                     dtype=x.dtype, device=x.device)
    
    # Calculate optimal block size
    BLOCK_SIZE = 16  # Can be tuned based on GPU architecture
    
    # Calculate grid dimensions
    grid = (batch_size, out_channels)
    
    # Launch kernel
    relu_maxpool_concat_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_relu_maxpool_concat