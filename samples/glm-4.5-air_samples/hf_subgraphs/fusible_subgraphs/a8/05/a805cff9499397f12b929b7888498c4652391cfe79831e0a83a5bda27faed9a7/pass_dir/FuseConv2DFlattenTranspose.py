import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Simple conv2d pattern - use positional arguments
    out = torch.conv2d(x, weight, bias, (16, 16), (0, 0), (1, 1), 1)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def optimized_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    height, width, kh, kw,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * out_channels * out_h * out_w)
    
    # Convert linear offset to multi-dimensional indices
    # [batch, out_channels, out_h, out_w]
    total_per_batch = out_channels * out_h * out_w
    batch_idx = offsets // total_per_batch
    remainder = offsets % total_per_batch
    
    channel_idx = remainder // (out_h * out_w)
    spatial_idx = remainder % (out_h * out_w)
    h_idx = spatial_idx // out_w
    w_idx = spatial_idx % out_w
    
    # Load input patch (simplified stride 16 convolution)
    # With stride 16, each output pixel comes from input at (h_idx*16, w_idx*16)
    input_val = 0.0
    kh_total = min(kh, 16)  # Since stride is 16, we only need up to 16x16
    kw_total = min(kw, 16)
    
    for kh_offset in range(kh_total):
        for kw_offset in range(kw_total):
            input_h = h_idx * 16 + kh_offset
            input_w = w_idx * 16 + kw_offset
            if input_h < height and input_w < width:
                input_idx = batch_idx * in_channels * height * width + \
                           channel_idx // (kw // kh) * height * width + \
                           input_h * width + input_w
                input_val += tl.load(x_ptr + input_idx, mask=(input_idx < batch_size * in_channels * height * width), other=0.0).to(tl.float32)
    
    # Load weights (simplified - sum over the kernel)
    weight_val = 0.0
    for kh_offset in range(min(kh, 16)):
        for kw_offset in range(min(kw, 16)):
            weight_idx = channel_idx * in_channels * kh * kw + \
                        0 * kh * kw + kh_offset * kw + kw_offset  # Assuming in_channels_per_group = 1
            weight_val += tl.load(weight_ptr + weight_idx, mask=(weight_idx < out_channels * in_channels * kh * kw), other=0.0).to(tl.float32)
    
    # Load bias
    bias_val = tl.load(bias_ptr + channel_idx, mask=(channel_idx < out_channels), other=0.0).to(tl.float32)
    
    # Compute output
    output_val = input_val * weight_val + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def optimized_conv2d(x, weight, bias):
    # Input shapes
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kh, kw = weight.shape
    
    # Calculate output dimensions for stride 16
    out_h = height // 16  # 224 // 16 = 14
    out_w = width // 16   # 224 // 16 = 14
    
    # Create output tensor: [batch, out_channels, out_h, out_w] = [1, 768, 14, 14]
    out_shape = (batch_size, out_channels, out_h, out_w)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Total output elements
    total_elements = batch_size * out_channels * out_h * out_w
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel = optimized_conv2d_kernel[num_programs]
    optimized_kernel(
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kh=kh,
        kw=kw,
        out_h=out_h,
        out_w=out_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_conv2d