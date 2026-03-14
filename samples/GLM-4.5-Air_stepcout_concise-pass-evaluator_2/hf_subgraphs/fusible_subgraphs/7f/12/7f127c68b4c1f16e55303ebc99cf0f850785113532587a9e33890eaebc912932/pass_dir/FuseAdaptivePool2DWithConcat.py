import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the computation pattern exactly as in the model:
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    
    The model only returns tmp_1, so we only return tmp_1 as observable output
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement
    """
    return (in_0, in_1)

@triton.jit
def fused_pool_concat_kernel(
    x_ptr,           # Input tensor ptr (in_0): [batch, 20, 64, 48]
    y_ptr,           # Input tensor ptr (in_1): [batch, 40, 32, 24]  
    out_ptr,         # Output ptr: [batch, 60, 32, 24]
    batch_size,
    in_channels_0,   # 20
    in_channels_1,   # 40
    out_channels,    # 60 (20 + 40)
    in_height,       # 64
    in_width,        # 48
    out_height,      # 32
    out_width,       # 24,
):
    """
    Fused kernel that performs:
    1. Average pooling with kernel_size=(2,2) and stride=(2,2)
    2. Concatenation along channel dimension
    """
    # Each program handles one output element (m, n, h) 
    m = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    w = 0  # We'll loop over width within each program
    
    # Bounds checking
    if m >= batch_size:
        return
    if n >= out_channels:
        return
    if h >= out_height:
        return
    
    # Loop over all width positions
    for w in range(out_width):
        # Determine if we're in the first half (pooled in_0) or second half (in_1)
        if n < in_channels_0:
            # First half: pooled from in_0
            # Calculate input coordinates (2x average pooling)
            in_h = h * 2
            in_w = w * 2
            
            # Load 2x2 neighborhood and compute average
            val00 = tl.load(x_ptr + m * in_channels_0 * in_height * in_width + 
                            n * in_height * in_width + in_h * in_width + in_w,
                            mask=(in_h < in_height) & (in_w < in_width), other=0.0)
            val01 = tl.load(x_ptr + m * in_channels_0 * in_height * in_width + 
                            n * in_height * in_width + in_h * in_width + (in_w + 1),
                            mask=(in_h < in_height) & (in_w + 1 < in_width), other=0.0)
            val10 = tl.load(x_ptr + m * in_channels_0 * in_height * in_width + 
                            n * in_height * in_width + (in_h + 1) * in_width + in_w,
                            mask=(in_h + 1 < in_height) & (in_w < in_width), other=0.0)
            val11 = tl.load(x_ptr + m * in_channels_0 * in_height * in_width + 
                            n * in_height * in_width + (in_h + 1) * in_width + (in_w + 1),
                            mask=(in_h + 1 < in_height) & (in_w + 1 < in_width), other=0.0)
            
            # Compute average
            pooled_val = (val00 + val01 + val10 + val11) * 0.25
            
        else:
            # Second half: directly from in_1
            y_channel = n - in_channels_0
            pooled_val = tl.load(y_ptr + m * in_channels_1 * out_height * out_width + 
                                y_channel * out_height * out_width + h * out_width + w,
                                mask=True, other=0.0)
        
        # Store the result
        offset = m * out_channels * out_height * out_width + \
                 n * out_height * out_width + \
                 h * out_width + w
        tl.store(out_ptr + offset, pooled_val)

@torch.fx.wrap  
def fused_pool_concat(in_0, in_1):
    """
    Fused function that performs adaptive pooling + concatenation
    """
    # Get input shapes
    batch_size, in_channels_0, in_height, in_width = in_0.shape
    _, in_channels_1, out_height, out_width = in_1.shape
    out_channels = in_channels_0 + in_channels_1
    
    # Validate input dimensions (should follow the expected pattern)
    assert in_height == 64, f"Expected input height 64, got {in_height}"
    assert in_width == 48, f"Expected input width 48, got {in_width}"
    assert out_height == 32, f"Expected output height 32, got {out_height}"
    assert out_width == 24, f"Expected output width 24, got {out_width}"
    assert in_channels_0 == 20, f"Expected in_0 channels 20, got {in_channels_0}"
    assert in_channels_1 == 40, f"Expected in_1 channels 40, got {in_channels_1}"
    
    # Create output tensor
    out_shape = (batch_size, out_channels, out_height, out_width)
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Launch kernel with 3D grid: (batch, channels, height)
    grid = (batch_size, out_channels, out_height)
    
    fused_pool_concat_kernel[grid](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        in_channels_0=in_channels_0,
        in_channels_1=in_channels_1,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
    )
    
    return out  # Return only the final concatenated result

def replacement_func():
    """
    Return the fused function as replacement
    """
    return fused_pool_concat