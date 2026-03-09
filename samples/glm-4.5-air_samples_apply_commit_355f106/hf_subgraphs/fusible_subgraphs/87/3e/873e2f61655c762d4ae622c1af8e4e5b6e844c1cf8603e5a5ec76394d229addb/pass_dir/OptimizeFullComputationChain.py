import torch
import triton
import triton.language as tl

def pattern(in_6, in_0):
    # Simple pattern: just match Conv2D
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return tmp_5

def replacement_args(in_6, in_0):
    return (in_6, in_0)

@triton.jit
def fused_1x1_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized program IDs for better GPU occupancy
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < spatial_size
    
    # Optimized memory layout for 1x1 convolution
    output_offset = (batch_id * out_channels + out_channel_id) * spatial_size + spatial_offset
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Vectorized reduction over input channels
    for in_c in range(in_channels):
        # Optimized memory access patterns
        weight_offset = out_channel_id * in_channels + in_c
        input_offset = (batch_id * in_channels + in_c) * spatial_size + spatial_offset
        
        # Direct loads without masking for weights (small tensors)
        weight_val = tl.load(weight_ptr + weight_offset).to(tl.float32)
        input_vals = tl.load(input_ptr + input_offset, mask=spatial_mask, other=0.0).to(tl.float32)
        
        # Accumulate with fused multiply-add
        acc += input_vals * weight_val
    
    # Store result with proper masking
    tl.store(output_ptr + output_offset, acc, mask=spatial_mask)

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, x + y, mask=mask)

@triton.jit
def fused_interpolate_add_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < output_height * output_width
    
    # Convert spatial offset to 2D coordinates
    output_y = spatial_offset // output_width
    output_x = spatial_offset % output_width
    
    # Calculate input coordinates and weights for bilinear interpolation
    x_scale = (input_width - 1) / max(output_width - 1, 1)
    y_scale = (input_height - 1) / max(output_height - 1, 1)
    
    input_x = output_x * x_scale
    input_y = output_y * y_scale
    
    # Bilinear interpolation setup
    x0 = tl.floor(input_x).to(tl.int32)
    y0 = tl.floor(input_y).to(tl.int32)
    x0 = tl.max(x0, 0).to(tl.int32)
    y0 = tl.max(y0, 0).to(tl.int32)
    x0 = tl.min(x0, input_width - 1).to(tl.int32)
    y0 = tl.min(y0, input_height - 1).to(tl.int32)
    
    fx = input_x - x0
    fy = input_y - y0
    
    x1 = tl.min(x0 + 1, input_width - 1).to(tl.int32)
    y1 = tl.min(y0 + 1, input_height - 1).to(tl.int32)
    
    # Input and output base addresses
    input_base = batch_id * channels * input_height * input_width + \
                 channel_id * input_height * input_width
    residual_base = batch_id * channels * output_height * output_width + \
                    channel_id * output_height * output_width
    output_base = residual_base
    
    # Calculate addresses for four corner points
    addr_00 = input_base + y0 * input_width + x0
    addr_01 = input_base + y1 * input_width + x0
    addr_10 = input_base + y0 * input_width + x1
    addr_11 = input_base + y1 * input_width + x1
    
    # Load four corner points
    q00 = tl.load(input_ptr + addr_00, mask=spatial_mask, other=0.0).to(tl.float32)
    q01 = tl.load(input_ptr + addr_01, mask=spatial_mask, other=0.0).to(tl.float32)
    q10 = tl.load(input_ptr + addr_10, mask=spatial_mask, other=0.0).to(tl.float32)
    q11 = tl.load(input_ptr + addr_11, mask=spatial_mask, other=0.0).to(tl.float32)
    
    # Bilinear interpolation
    top = (1 - fx) * q00 + fx * q10
    bottom = (1 - fx) * q01 + fx * q11
    interpolated = (1 - fy) * top + fy * bottom
    
    # Load residual value
    residual_addr = residual_base + spatial_offset
    residual_val = tl.load(residual_ptr + residual_addr, mask=spatial_mask, other=0.0).to(tl.float32)
    
    # Add residual
    result = interpolated + residual_val
    
    tl.store(output_ptr + output_base + spatial_offset, result, mask=spatial_mask)

@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels,
    height,
    width,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offset < width
    
    input_base = batch_id * channels * height * width + \
                 channel_id * height * width + \
                 spatial_offset
    
    # Load batch norm parameters
    mean_val = tl.load(mean_ptr + channel_id).to(tl.float32)
    var_val = tl.load(var_ptr + channel_id).to(tl.float32)
    weight_val = tl.load(weight_ptr + channel_id).to(tl.float32)
    bias_val = tl.load(bias_ptr + channel_id).to(tl.float32)
    
    # Load input and compute fused BatchNorm + ReLU
    input_val = tl.load(input_ptr + input_base, mask=spatial_mask, other=0.0).to(tl.float32)
    
    # BatchNorm computation
    normalized = (input_val - mean_val) / tl.sqrt(var_val + 1e-05)
    batch_norm_out = weight_val * normalized + bias_val
    
    # ReLU activation
    relu_out = tl.where(batch_norm_out > 0, batch_norm_out, 0.0)
    
    tl.store(output_ptr + input_base, relu_out, mask=spatial_mask)

@torch.fx.wrap
def simple_optimized_conv(in_6, in_0):
    # Simple 1x1 convolution optimization
    batch_size, in_channels, input_h, input_w = in_6.shape
    out_channels = in_0.shape[0]
    
    conv_out = torch.empty(batch_size, out_channels, input_h, input_w,
                          device=in_6.device, dtype=in_6.dtype)
    
    grid_z = batch_size
    grid_y = out_channels
    # Use larger block size for better GPU occupancy
    spatial_elements = input_h * input_w
    block_size = 256 if spatial_elements >= 512 else 128  # Adaptive block size
    grid_x = (spatial_elements + block_size - 1) // block_size
    
    fused_1x1_conv_kernel[(
        grid_z, grid_y, grid_x
    )](
        in_6, in_0, conv_out,
        batch_size, in_channels, out_channels,
        spatial_elements, block_size
    )
    
    return conv_out

def replacement_func():
    return simple_optimized_conv