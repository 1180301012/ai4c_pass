import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D + HardSwish fusion
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Fused Conv2D + HardSwish kernel - Simplified and robust version
@triton.jit
def fused_conv_hardswish_kernel_simplified(
    input_ptr,           # Input tensor
    weight_ptr,          # Weight tensor  
    bias_ptr,            # Bias tensor
    output_ptr,          # Output tensor
    batch_size,          # Batch size
    in_channels,         # Input channels
    out_channels,        # Output channels
    input_height,        # Input height
    input_width,         # Input width,
    stride_h,            # Stride height
    stride_w,            # Stride width
    dilation_h,          # Dilation height
    dilation_w,          # Dilation width
    CHANNEL_BLOCK_SIZE: tl.constexpr,    # Block size for processing channels
):
    # Each program handles one element in the output
    m = tl.program_id(0)  # Output channel dimension
    n = tl.program_id(1)  # Batch * spatial dimension
    
    # Compute offsets
    batch = n // (input_height * input_width)
    spatial_idx = n % (input_height * input_width)
    h_idx = spatial_idx // input_width
    w_idx = spatial_idx % input_width
    
    # Compute output coordinates (for 1x1 conv with stride 1, output == input)
    out_h = h_idx * stride_h // dilation_h
    out_w = w_idx * stride_w // dilation_w
    
    # Early return for out-of-bounds access - avoid chained boolean operators
    out_of_bounds = False
    if m >= out_channels:
        out_of_bounds = True
    if not out_of_bounds and out_h >= input_height:
        out_of_bounds = True
    if not out_of_bounds and out_w >= input_width:
        out_of_bounds = True
    if out_of_bounds:
        return
    
    # Load bias for this output channel
    result = tl.load(bias_ptr + m)
    
    # Process channels in blocks for better memory locality
    for k_offset in range(0, in_channels, CHANNEL_BLOCK_SIZE):
        # Determine the actual range for this block
        k_end = min(k_offset + CHANNEL_BLOCK_SIZE, in_channels)
        
        # Process each channel in the current block
        # Use individual loads to avoid dynamic arange issues
        for k in range(k_offset, k_end):
            # Calculate weight offset: [out_channels, in_channels] layout
            weight_offset = m * in_channels + k
            
            # Calculate input offset: [batch, in_channels, H, W] layout
            input_offset = batch * in_channels * input_height * input_width + \
                          k * input_height * input_width + \
                          out_h * input_width + out_w
            
            # Load weight and input values
            weight_val = tl.load(weight_ptr + weight_offset)
            input_val = tl.load(input_ptr + input_offset)
            
            # Accumulate: result += weight * input
            result += weight_val * input_val
    
    # Apply HardSwish activation: hardswish(x) = x * relu6(x + 3) / 6
    # Optimized fused operations
    relu6_input = result + 3.0
    relu6_output = tl.minimum(tl.maximum(relu6_input, 0.0), 6.0)
    final_output = result * relu6_output / 6.0
    
    # Store result
    output_offset = n * out_channels + m
    tl.store(output_ptr + output_offset, final_output)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_hardswish_kernel_wrapper(in_0, in_1, in_2):
    # Get input shapes
    in_2_shape = in_2.shape
    batch_size, in_channels, input_height, input_width = in_2_shape
    
    # Get weight shape  
    out_channels = in_1.shape[0]
    
    # Calculate output size (for 1x1 conv with stride 1)
    output_height = input_height
    output_width = input_width
    
    # Initialize output tensor
    output_shape = (batch_size, out_channels, output_height, output_width)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration - use moderate block sizes for better GPU occupancy
    CHANNEL_BLOCK_SIZE = 32  # Process 32 channels at a time (balance between vectorization and memory)
    
    # Calculate grid dimensions
    m_tiles = (out_channels + CHANNEL_BLOCK_SIZE - 1) // CHANNEL_BLOCK_SIZE
    n_tiles = batch_size * input_height * input_width
    
    grid_size = (m_tiles, n_tiles)
    
    # Launch simplified kernel
    fused_conv_hardswish_kernel_simplified[grid_size](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        CHANNEL_BLOCK_SIZE=CHANNEL_BLOCK_SIZE
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv_hardswish_kernel_wrapper