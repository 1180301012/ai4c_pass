import torch
import triton
import triton.language as tl

# Pattern matching function for both operations in the model
def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match both Conv2D and Add+Dropout operations from the model"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    tmp_3 = None
    return (tmp_4, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

# Optimized convolution kernel
@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Decode batch, output_channel, h, w from linear index
    linear_idx = pid
    if linear_idx >= batch_size * out_channels * height * width:
        return
        
    # Calculate coordinates
    b = linear_idx // (out_channels * height * width)
    remainder = linear_idx % (out_channels * height * width)
    c_out = remainder // (height * width)
    remainder2 = remainder % (height * width)
    h = remainder2 // width
    w = remainder2 % width
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + c_out)
    
    # Compute convolution: bias[c_out] + sum_{c_in} (input[b, c_in, h, w] * weight[c_out, c_in])
    result = bias_val
    
    # Loop through input channels
    for c_in in range(in_channels):
        # Compute input tensor index: (b, c_in, h, w)
        input_idx = (b * in_channels + c_in) * height * width + h * width + w
        input_val = tl.load(input_ptr + input_idx)
        
        # Compute weight tensor index: (c_out, c_in)
        weight_idx = c_out * in_channels + c_in
        weight_val = tl.load(weight_ptr + weight_idx)
        
        # Add to result
        result += input_val * weight_val
    
    # Store the result
    tl.store(output_ptr + linear_idx, result)

# Fused addition kernel (dropout becomes identity when training=False)
@triton.jit
def add_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for spatial blocks
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    spatial_elements = height * width
    
    # Each program handles a block of spatial elements
    block_size = BLOCK_SIZE * channels  # Process multiple channels per block
    block_start = pid * block_size
    block_end = min(block_start + block_size, total_elements)
    
    if block_start >= total_elements:
        return
        
    # Process elements in the block
    for idx in range(block_start, block_end):
        # Decode coordinates
        b = idx // (channels * height * width)
        remainder = idx % (channels * height * width)
        c = remainder // (height * width)
        remainder2 = remainder % (height * width)
        h = remainder2 // width
        w = remainder2 % width
        
        # Load input values
        x1_idx = (b * channels + c) * height * width + h * width + w
        x2_idx = (b * channels + c) * height * width + h * width + w
        
        x1_val = tl.load(x1_ptr + x1_idx)
        x2_val = tl.load(x2_ptr + x2_idx)
        
        # Compute sum
        result = x1_val + x2_val
        tl.store(output_ptr + idx, result)

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2, in_3, in_4):
    """Optimized computation combining both operations"""
    
    # Conv2D operation
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_1.shape[0]
    
    # Create output tensor for conv2d
    conv_output = torch.empty((batch_size, out_channels, height, width), 
                             dtype=in_2.dtype, device=in_2.device)
    
    # Launch conv2d kernel
    total_elements = batch_size * out_channels * height * width
    num_programs = (total_elements + 255 - 1) // 256
    grid = (num_programs,)
    
    conv2d_kernel[grid](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=conv_output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width
    )
    
    # Addition + Dropout operation (fused as just addition since dropout=False)
    x1_shape = in_4.shape
    BATCH_SIZE = 256
    
    output_add = torch.empty_like(in_4)
    
    grid = ((in_4.numel() + BATCH_SIZE - 1) // BATCH_SIZE,)
    add_kernel[grid](
        x1_ptr=in_4,
        x2_ptr=in_3,
        output_ptr=output_add,
        batch_size=x1_shape[0],
        channels=x1_shape[1],
        height=x1_shape[2],
        width=x1_shape[3],
        BLOCK_SIZE=BATCH_SIZE
    )
    
    return (output_add, conv_output)

# Replacement function
def replacement_func():
    return optimized_forward