import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match adaptive_avg_pool2d followed by concat along dim=1"""
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)

def replacement_args(in_0, in_1):
    """Extract arguments for fused operation"""
    return (in_0, in_1)

@triton.jit
def fused_pool_concat_kernel(
    x_ptr,               # Input tensor to be pooled (in_0)
    y_ptr,               # Tensor to concatenate (in_1) 
    out_ptr,             # Output tensor
    n_batch,             # Batch size
    n_channels_x,        # Channels in in_0 (20)
    n_channels_y,        # Channels in in_1 (40)
    in_height,           # Input height (64)
    in_width,            # Input width (48)
    out_height,          # Output height (32)
    out_width,           # Output width (24)
):
    """Fused adaptive average pooling + concatenation kernel"""
    
    # Grid coordinates - each program handles one spatial position
    batch_id = tl.program_id(0).to(tl.int32)
    out_y = tl.program_id(1).to(tl.int32) 
    out_x = tl.program_id(2).to(tl.int32)
    
    # Input region mapping for adaptive pooling
    y_start = (out_y * in_height) // out_height
    y_end = ((out_y + 1) * in_height) // out_height
    y_end = tl.minimum(y_end, in_height)
    
    x_start = (out_x * in_width) // out_width
    x_end = ((out_x + 1) * in_width) // out_width  
    x_end = tl.minimum(x_end, in_width)
    
    # Batch and spatial masks
    batch_mask = batch_id < n_batch
    spatial_mask = (out_y < out_height) & (out_x < out_width)
    
    if not (spatial_mask & batch_mask):
        return
    
    # Loop through all output channels
    for ch_idx in range(n_channels_x + n_channels_y):
        
        # Skip if batch is out of bounds
        if batch_mask:
            
            # Base address for this batch and channel in output
            out_base = (batch_id * (n_channels_x + n_channels_y) + ch_idx) * out_height * out_width
            out_pos = out_base + out_y * out_width + out_x
            
            # For channels in in_1 (direct copy part)
            if ch_idx >= n_channels_x:
                ch_y_idx = ch_idx - n_channels_x
                
                # Calculate source position in y tensor
                y_base = (batch_id * n_channels_y + ch_y_idx) * out_height * out_width
                y_pos = y_base + out_y * out_width + out_x
                
                # Load and copy
                y_value = tl.load(y_ptr + y_pos, mask=batch_mask, other=0.0)
                tl.store(out_ptr + out_pos, y_value, mask=batch_mask)
            
            # For channels in in_0 (adaptive pooling part)
            else:
                ch_x_idx = ch_idx
                
                # Calculate source addresses based from one element
                x_base = (batch_id * n_channels_x + ch_x_idx) * in_height * in_width
                
                # Sum over input region for average pooling
                total = 0.0
                count = 0
                
                # Loop over input region
                for y_in in range(y_start, y_end):
                    for x_in in range(x_start, x_end):
                        # Calculate input position
                        in_pos = x_base + y_in * in_width + x_in
                        
                        # Load input value
                        in_value = tl.load(x_ptr + in_pos, mask=batch_mask, other=0.0)
                        total += in_value
                        count += 1
                
                # Compute average and store
                if count > 0:
                    avg_value = total / count
                    tl.store(out_ptr + out_pos, avg_value, mask=batch_mask)

@torch.fx.wrap 
def fused_pool_concat_function(in_0, in_1):
    """Wrapper function for fused adaptive pooling + concatenation"""
    
    # Get input shapes
    n_batch, n_channels_x, in_height, in_width = in_0.shape
    n_channels_y = in_1.shape[1]
    out_height, out_width = 32, 24
    
    # Output shape: [batch, n_channels_x + n_channels_y, 32, 24]
    n_channels_out = n_channels_x + n_channels_y
    output_shape = (n_batch, n_channels_out, out_height, out_width)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration: each program handles one (batch, out_y, out_x) position
    grid_z = n_batch
    grid_y = out_height  
    grid_x = out_width
    
    # Launch kernel - each thread processes one spatial position for all channels
    fused_pool_concat_kernel[(grid_z, grid_y, grid_x)](
        in_0, in_1, output,
        n_batch, n_channels_x, n_channels_y,
        in_height, in_width,
        out_height, out_width
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_pool_concat_function