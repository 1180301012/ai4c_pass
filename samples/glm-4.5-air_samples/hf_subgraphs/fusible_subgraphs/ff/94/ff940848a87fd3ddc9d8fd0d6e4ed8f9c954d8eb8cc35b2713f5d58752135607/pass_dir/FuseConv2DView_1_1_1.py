import torch
import triton
import triton.language as tl

# Pattern matching function - just conv2d operation only
def pattern(x, w):
    # Simple conv2d operation
    return torch.conv2d(x, w, None, (1, 1), (0, 0), (1, 1), 1)

# Argument extraction function
def replacement_args(x, w):
    return (x, w)

# Optimized kernel for conv2d + view fusion
@triton.jit
def fused_conv2d_view_kernel(
    conv_in_ptr,
    conv_weight_ptr, 
    conv_bias_ptr,
    view_out_ptr,
    n_elements,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Determine the total elements per batch and per feature map
    elements_per_batch = height * width
    total_elements_per_feature_map = batch_size * elements_per_batch
    
    # Calculate conv2d output: (batch, out_channels, height, width) -> (batch, out_channels, -1)
    # We need to flatten the spatial dimensions (height, width) into (-1)
    
    # For each output position in the flattened tensor
    feature_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Calculate batch index and feature index
    batch_idx = spatial_idx // (out_channels * height * width)
    out_channel_idx = (spatial_idx // (height * width)) % out_channels
    spatial_flat_idx = spatial_idx % (height * width)
    
    # Spatial coordinates
    h = spatial_flat_idx // width
    w = spatial_flat_idx % width
    
    # Flatten the spatial dimension for output
    output_flat_idx = batch_idx * out_channels * height * width + feature_idx * height * width + spatial_flat_idx
    
    if output_flat_idx < n_elements:
        # Load bias
        bias_val = tl.load(conv_bias_ptr + feature_idx)
        
        # Load weight (assuminggroups=1, weights are [out_channels, in_channels, 1, 1])
        weight_val = tl.load(conv_weight_ptr + feature_idx * in_channels)
        
        # Load input: [batch_size, in_channels, height, width]
        input_val = tl.load(conv_in_ptr + batch_idx * in_channels * height * width + 
                           feature_idx * height * width + h * width + w)
        
        # Conv2D with stride 1, padding 0, dilation 1, groups=1
        # For 1x1 convolution, this reduces to simple element-wise operations
        conv_val = input_val * weight_val + bias_val
        
        # Store in the flattened view output
        tl.store(view_out_ptr + output_flat_idx, conv_val, mask=output_flat_idx < n_elements)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv2d_view_wrapper(conv_in, conv_weight, conv_bias, view_out):
    # Calculate shapes
    batch_size, in_channels, height, width = conv_in.shape
    out_channels = conv_weight.shape[0]
    
    # Create output tensor for fused operation (viewed as [batch_size, out_channels, -1])
    output_shape = (batch_size, out_channels, height * width)
    output_size = batch_size * out_channels * height * width
    
    out = torch.empty(output_shape, dtype=conv_in.dtype, device=conv_in.device)
    
    # Calculate block size and grid dimensions
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For 1x1 convolution, we can simplify the grid dimensions
    grid = (
        num_programs,
        out_channels,
        batch_size,
    )
    
    # Launch kernel
    fused_conv2d_view_kernel[grid](
        conv_in_ptr=conv_in,
        conv_weight_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        view_out_ptr=out,
        n_elements=output_size,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Optimized kernel for fused 1x1 conv2d + view operation
@triton.jit
def fused_conv2d_view_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * height * width
    
    # Calculate indices
    batch_idx = offsets // (out_channels * height * width)
    out_channel_idx = (offsets // (height * width)) % out_channels
    spatial_flat_idx = offsets % (height * width)
    h_idx = spatial_flat_idx // width
    w_idx = spatial_flat_idx % width
    
    # Calculate base addresses
    input_base = batch_idx * in_channels * height * width + out_channel_idx * height * width + h_idx * width + w_idx
    weight_base = out_channel_idx * in_channels  # For 1x1 convolution
    bias_base = out_channel_idx
    output_base = offsets  # Since we're flattening directly
    
    # Load data with bounds checking
    x = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    w = tl.load(weight_ptr + weight_base, mask=mask, other=0.0)
    b = tl.load(bias_ptr + bias_base, mask=mask, other=0.0)
    
    # Perform 1x1 convolution (simplified to element-wise operations)
    result = x * w + b
    
    # Store directly to flattened output
    tl.store(output_ptr + output_base, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv2d_view_wrapper(in_3, in_1, in_0):
    # Get tensor shapes
    batch_size, in_channels, height, width = in_3.shape
    out_channels = in_1.shape[0]  # [256, 512, 1, 1] -> out_channels = 256
    
    # Create output tensor directly in the desired shape
    output_shape = (batch_size, out_channels, height * width)
    output = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Setup kernel launch parameters
    BLOCK_SIZE = 1024
    num_elements = batch_size * out_channels * height * width
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv2d_view_kernel[(num_programs,)](
        input_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Optimized 1x1 conv2d kernel using Triton
@triton.jit
def conv1x1_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * height * width
    
    # Calculate indices
    batch_idx = offsets // (out_channels * height * width)
    out_channel_idx = (offsets // (height * width)) % out_channels
    spatial_flat_idx = offsets % (height * width)
    h_idx = spatial_flat_idx // width
    w_idx = spatial_flat_idx % width
    
    # For 1x1 conv2d: element-wise multiplication and reduction over input channels
    # This is a simplified version that assumes weight shape [out_channels, in_channels, 1, 1]
    result = 0.0
    
    # For each input channel (reduction)
    for in_channel_idx in tl.static_range(in_channels):
        # Calculate input address
        input_base = batch_idx * in_channels * height * width + in_channel_idx * height * width + h_idx * width + w_idx
        weight_base = out_channel_idx * in_channels + in_channel_idx
        
        # Load input and weight
        x = tl.load(x_ptr + input_base, mask=mask)
        w = tl.load(w_ptr + weight_base, mask=mask)
        
        result += x * w
    
    # Store result (no bias for simplicity, can be added later)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_conv2d_wrapper(x, w):
    # Get tensor shapes
    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Setup kernel launch parameters
    BLOCK_SIZE = 1024
    num_elements = batch_size * out_channels * height * width
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv1x1_kernel[(num_programs,)](
        x_ptr=x,
        w_ptr=w,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_conv2d_wrapper