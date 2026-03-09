import torch
import triton
import triton.language as tl

# Pattern matching function for full Conv2D + HardSwish + Flatten fusion
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    tmp_2 = None
    tmp_4 = tmp_3.flatten(1, -1)
    tmp_3 = None
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Fused Conv2D + HardSwish + Flatten kernel - High-performance version
@triton.jit
def fused_conv_hardswish_flatten_kernel_final(
    input_ptr,           # Input tensor
    weight_ptr,          # Weight tensor  
    bias_ptr,            # Bias tensor
    output_ptr,          # Flattened output tensor
    batch_size,          # Batch size
    in_channels,         # Input channels
    out_channels,        # Output channels
    input_height,        # Input height
    input_width,         # Input width,
    OUTPUT_BLOCK_SIZE: tl.constexpr,    # Block size for output channels
):
    # Each program handles one output channel for all spatial positions in a batch
    m = tl.program_id(0)  # Output channel dimension
    batch_spatial_idx = tl.program_id(1)  # Batch * spatial index
    
    # Extract batch and spatial coordinates
    batch_idx = batch_spatial_idx // (input_height * input_width)
    spatial_idx = batch_spatial_idx % (input_height * input_width)
    h_idx = spatial_idx // input_width
    w_idx = spatial_idx % input_width
    
    # Early return for out-of-bounds access - avoid chained boolean operators
    out_of_bounds = False
    if m >= out_channels:
        out_of_bounds = True
    if not out_of_bounds and batch_idx >= batch_size:
        out_of_bounds = True
    if not out_of_bounds and h_idx >= input_height:
        out_of_bounds = True
    if not out_of_bounds and w_idx >= input_width:
        out_of_bounds = True
    if out_of_bounds:
        return
    
    # Load bias for this output channel
    result = tl.load(bias_ptr + m)
    
    # Vectorized 1x1 convolution: process ALL input channels at once for efficiency
    # For 1x1 conv at position (h_idx, w_idx), each output channel computes:
    # sum_{k=0}^{in_channels-1} weight[m, k] * input[batch_idx, k, h_idx, w_idx]
    
    # Calculate base offset for input tensor at this spatial position
    input_base = batch_idx * in_channels * input_height * input_width + \
                 h_idx * input_width + w_idx
    
    # Calculate base offset for weight tensor for this output channel
    weight_base = m * in_channels
    
    # Process input channels efficiently (no blocking for small 1x1 conv)
    # Vectorized access pattern optimized for 1x1 convolution
    for k in range(0, in_channels, OUTPUT_BLOCK_SIZE):
        k_end = min(k + OUTPUT_BLOCK_SIZE, in_channels)
        
        # Create compile-time constant ranges
        k_range = tl.arange(k, k_end)
        
        # Vectorized memory access
        weight_offsets = weight_base + k_range
        input_offsets = input_base + k_range * input_height * input_width
        
        # Load vectors
        weights = tl.load(weight_ptr + weight_offsets)
        inputs = tl.load(input_ptr + input_offsets)
        
        # Vectorized computation for this channel block
        result += tl.sum(weights * inputs)
    
    # Apply HardSwish activation: hardswish(x) = x * relu6(x + 3) / 6
    # Optimized fused operations - minimize operations
    relu6_input = result + 3.0
    relu6_output = tl.minimum(tl.maximum(relu6_input, 0.0), 6.0)
    final_output = result * relu6_output / 6.0
    
    # Store result in flattened format: [batch * spatial, channels]
    output_offset = batch_spatial_idx * out_channels + m
    tl.store(output_ptr + output_offset, final_output)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_hardswish_flatten_kernel_wrapper(in_0, in_1, in_2):
    # Get input shapes
    in_2_shape = in_2.shape
    batch_size, in_channels, input_height, input_width = in_2_shape
    
    # Get weight shape  
    out_channels = in_1.shape[0]
    
    # Calculate flattened output shape: [batch * spatial, channels]
    flattened_size = batch_size * input_height * input_width * out_channels
    
    # Initialize flattened output tensor
    output = torch.empty(flattened_size, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration optimized for 1x1 convolution
    # Each program handles one output channel for one spatial position across all batches
    OUTPUT_BLOCK_SIZE = 16  # Vector size for processing input channels
    
    # Calculate grid dimensions:
    # - First dimension: output channels
    # - Second dimension: batch * spatial positions  
    m_tiles = (out_channels + OUTPUT_BLOCK_SIZE - 1) // OUTPUT_BLOCK_SIZE
    n_tiles = batch_size * input_height * input_width
    
    grid_size = (m_tiles, n_tiles)
    
    # Launch final optimized kernel
    fused_conv_hardswish_flatten_kernel_final[grid_size](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        OUTPUT_BLOCK_SIZE=OUTPUT_BLOCK_SIZE
    )
    
    return (output,)

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv_hardswish_flatten_kernel_wrapper