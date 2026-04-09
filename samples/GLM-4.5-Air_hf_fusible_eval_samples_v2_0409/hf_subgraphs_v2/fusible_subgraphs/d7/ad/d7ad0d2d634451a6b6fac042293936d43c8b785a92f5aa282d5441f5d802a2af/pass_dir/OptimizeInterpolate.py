import torch
import triton
import triton.language as tl

def pattern(x):
    interpolated = torch.nn.functional.interpolate(
        x, size=(128, 128), mode='bilinear', align_corners=False
    )
    return interpolated

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_interpolate_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_y = tl.program_id(2)  # output height
    pid_x = tl.program_id(3)  # output width
    
    # Calculate output coordinates
    out_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    out_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Bounds checking
    y_mask = out_y < out_height
    x_mask = out_x < out_width
    
    # Scale factors for bilinear interpolation
    y_scale = (in_height - 1) / (out_height - 1)
    x_scale = (in_width - 1) / (out_width - 1)
    
    # Calculate input coordinates
    in_y = out_y * y_scale
    in_x = out_x * x_scale
    
    # Floor coordinates for interpolation
    y0 = tl.floor(in_y)
    x0 = tl.floor(in_x)
    y1 = y0 + 1
    x1 = x0 + 1
    
    # Bounds checking for input coordinates
    y0_mask = y0 < in_height
    x0_mask = x0 < in_width
    y1_mask = y1 < in_height
    x1_mask = x1 < in_width
    
    # Calculate interpolation weights
    wy1 = in_y - y0
    wx1 = in_x - x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    
    # Clamp coordinates to valid range
    y0 = tl.maximum(y0, 0)
    x0 = tl.maximum(x0, 0)
    y1 = tl.minimum(y1, in_height - 1)
    x1 = tl.minimum(x1, in_width - 1)
    
    # Load four corner pixels using vectorized loads
    input_data = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Load top-left pixel (y0, x0)
    if pid_b < batch_size and pid_c < channels and tl.any(y_mask & x_mask):
        offset = (pid_b * channels + pid_c) * in_height * in_width + \
                 y0.reshape(-1, 1) * in_width + x0.reshape(1, -1)
        input_data = tl.load(x_ptr + offset, mask=y_mask.reshape(-1, 1) & x_mask.reshape(1, -1), other=0.0)
    
    # Load top-right pixel (y0, x1)
    if pid_b < batch_size and pid_c < channels and tl.any(y_mask & x_mask):
        offset = (pid_b * channels + pid_c) * in_height * in_width + \
                 y0.reshape(-1, 1) * in_width + x1.reshape(1, -1)
        top_right = tl.load(x_ptr + offset, mask=y_mask.reshape(-1, 1) & x1_mask.reshape(1, -1), other=0.0)
        tensor_wx0 = wx0.reshape(-1, 1)
        input_data = input_data * tensor_wx0 + top_right * (1.0 - tensor_wx0)
    
    # Load bottom-left pixel (y1, x0)
    if pid_b < batch_size and pid_c < channels and tl.any(y_mask & x_mask):
        offset = (pid_b * channels + pid_c) * in_height * in_width + \
                 y1.reshape(-1, 1) * in_width + x0.reshape(1, -1)
        bottom_left = tl.load(x_ptr + offset, mask=y1_mask.reshape(-1, 1) & x_mask.reshape(1, -1), other=0.0)
        tensor_wy0 = wy0.reshape(-1, 1)
        input_data = input_data * tensor_wy0 + bottom_left * (1.0 - tensor_wy0)
    
    # Load bottom-right pixel (y1, x1)
    if pid_b < batch_size and pid_c < channels and tl.any(y_mask & x_mask):
        offset = (pid_b * channels + pid_c) * in_height * in_width + \
                 y1.reshape(-1, 1) * in_width + x1.reshape(1, -1)
        bottom_right = tl.load(x_ptr + offset, mask=y1_mask.reshape(-1, 1) & x1_mask.reshape(1, -1), other=0.0)
        tensor_wy1_tensor_wx1 = wy1.reshape(-1, 1) * (1.0 - wx0.reshape(1, -1))
        input_data = input_data * tensor_wy0.reshape(-1, 1) + bottom_right * tensor_wy1_tensor_wx1
    
    # Store result
    out_offset = (pid_b * channels + pid_c) * out_height * out_width + \
                out_y.reshape(-1, 1) * out_width + out_x.reshape(1, -1)
    
    tl.store(out_ptr + out_offset, 
             input_data, 
             mask=y_mask.reshape(-1, 1) & x_mask.reshape(1, -1))

@torch.fx.wrap
def optimized_interpolate(x):
    batch_size, channels, in_height, in_width = x.shape
    out_height = 128
    out_width = 128
    
    # Output tensor
    out = torch.empty((batch_size, channels, out_height, out_width), 
                     device=x.device, dtype=x.dtype)
    
    # Triton kernel config
    BLOCK_SIZE_Y = 16
    BLOCK_SIZE_X = 16
    
    # Calculate grid dimensions
    grid_y = (out_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (out_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid = (batch_size, channels, grid_y, grid_x)
    
    # Launch optimized kernel
    optimized_interpolate_kernel[grid, (BLOCK_SIZE_Y, BLOCK_SIZE_X)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_X=BLOCK_SIZE_X
    )
    
    return out

def replacement_func():
    return optimized_interpolate