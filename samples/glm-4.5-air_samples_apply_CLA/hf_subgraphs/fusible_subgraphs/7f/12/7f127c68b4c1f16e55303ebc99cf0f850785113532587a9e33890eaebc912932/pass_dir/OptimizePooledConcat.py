import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the target computation
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Simple optimized kernel that handles pooling with proper memory layout
@triton.jit
def pooled_concat_kernel(
    x_ptr,           # Input tensor in_0: [B, C_in, H_in, W_in] 
    y_ptr,           # Input tensor in_1: [B, C_y, H_out, W_out]
    out_ptr,         # Output tensor: [B, C_in+C_y, H_out, W_out]
    batch_size,      
    c_in,            # Channels for in_0 (20)
    c_y,             # Channels for in_1 (40)
    h_in,            # Input height (64)
    w_in,            # Input width (48)
    h_out,           # Output height (32) 
    w_out,           # Output width (24)
):
    # Each program handles ONE output element (batch, channel, height, width)
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Check bounds - separate conditions to avoid chained boolean operators
    if (batch_idx >= batch_size):
        return
    if (channel_idx >= (c_in + c_y)):
        return
    if (spatial_idx >= (h_out * w_out)):
        return
    
    # Calculate spatial coordinates
    out_h = spatial_idx // w_out
    out_w = spatial_idx % w_out
    
    # Check spatial bounds again - separate conditions
    if (out_h >= h_out):
        return
    if (out_w >= w_out):
        return
    
    if channel_idx < c_in:
        # This is from in_0 - perform average pooling
        src_c = channel_idx
        
        # Calculate source spatial coordinates for 2x2 pooling
        src_h = out_h * 2
        src_w = out_w * 2
        
        # Sum over 2x2 neighborhood, handling edge cases
        sum_val = 0.0
        count = 0
        for dy in range(2):
            for dx in range(2):
                h_idx = src_h + dy
                w_idx = src_w + dx
                if h_idx < h_in and w_idx < w_in:
                    # NCHW layout: offset = batch * (C*H*W) + channel * (H*W) + h * W + w
                    offset = batch_idx * c_in * h_in * w_in + src_c * h_in * w_in + h_idx * w_in + w_idx
                    val = tl.load(x_ptr + offset)
                    sum_val += val
                    count += 1
        
        value = sum_val / count if count > 0 else 0.0
    else:
        # This is from in_1 - direct copy
        src_c = channel_idx - c_in
        # NCHW layout: offset = batch * (C*H*W) + channel * (H*W) + h * W + w
        offset = batch_idx * c_y * h_out * w_out + src_c * h_out * w_out + out_h * w_out + out_w
        value = tl.load(y_ptr + offset)
    
    # Store result in NCHW layout
    # offset = batch * (C_out*H_out*W_out) + channel * (H_out*W_out) + h * W_out + w
    out_offset = batch_idx * (c_in + c_y) * h_out * w_out + channel_idx * h_out * w_out + out_h * w_out + out_w
    tl.store(out_ptr + out_offset, value)

# Wrapper function with Triton integration
@torch.fx.wrap
def triton_pooled_concat(in_0, in_1):
    # Get input shapes
    batch_size, c_in, h_in, w_in = in_0.shape
    _, c_y, h_out, w_out = in_1.shape
    
    # Validate that pooling dimensions are as expected
    assert h_out == h_in // 2, f"Output height should be half input height: {h_in//2} vs {h_out}"
    assert w_out == w_in // 2, f"Output width should be half input width: {w_in//2} vs {w_out}"
    assert h_out == 32, f"Expected output height 32, got {h_out}"
    assert w_out == 24, f"Expected output width 24, got {w_out}"
    
    # Create output tensor
    out = torch.empty((batch_size, c_in + c_y, h_out, w_out), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid dimensions: 3D grid (batch, channel, spatial)
    grid_z = batch_size
    grid_y = c_in + c_y
    grid_x = h_out * w_out  # Combine spatial dimensions for better efficiency
    
    # Launch kernel with 3D grid
    pooled_concat_kernel[(grid_z, grid_y, grid_x)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        c_in=c_in,
        c_y=c_y,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
    )
    
    return out

# Replacement function that returns the optimized implementation
def replacement_func():
    return triton_pooled_concat