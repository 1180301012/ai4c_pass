import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern to match Conv2D + Hardswish + Flatten operations
    in_0: bias tensor [1280]
    in_1: weight tensor [1280, 960, 1, 1]
    in_2: input tensor [batch, 960, 1, 1]
    """
    # Conv2D with given parameters
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Hardswish activation (inplace=True)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    # Flatten from channel dimension
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel"""
    return in_0, in_1, in_2

@triton.jit
def fused_conv_hardswish_flatten_kernel(
    output_ptr,
    bias_ptr,
    weight_ptr,
    input_ptr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
):
    """Optimized fused kernel for 1x1 Conv2D + Hardswish + Flatten
    
    Each program processes one output channel for a specific batch.
    This kernel fuses three operations:
    1. 1x1 convolution (equivalent to matrix multiplication)
    2. Hardswish activation: hardswish(x) = x * relu6(x + 3) / 6
    3. Flatten from [batch, output_channels, 1, 1] to [batch, output_channels]
    """
    # Each program handles one output channel for one batch
    output_channel = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + output_channel)
    
    # Compute the matrix multiplication: bias + sum(weight * input)
    output_val = bias
    
    # Load input for this batch: [input_channels]
    input_offset = batch_idx * input_channels
    
    # Compute output for this output channel with optimized memory access
    for ic in range(0, input_channels, 128):
        # Always load 128 elements, mask handles bounds
        indices = tl.arange(0, 128)
        mask = indices < (input_channels - ic)
        
        # Load weight and input blocks
        weight_offset = output_channel * input_channels + ic
        weight_block = tl.load(weight_ptr + weight_offset + indices, mask=mask)
        input_block = tl.load(input_ptr + input_offset + ic + indices, mask=mask)
        
        # Matrix multiply: output += weight * input_elemwise
        sum_val = tl.sum(weight_block * input_block)
        output_val += tl.cast(sum_val, output_val.dtype)
    
    # Apply hardswish activation: x * relu6(x + 3) / 6
    x_plus_3 = output_val + 3
    relu6_result = tl.maximum(tl.minimum(x_plus_3, 6.0), 0.0)
    hardswish_result = output_val * relu6_result / 6.0
    
    # Store result in flattened output: [batch, output_channels]
    output_offset = batch_idx * output_channels + output_channel
    tl.store(output_ptr + output_offset, hardswish_result)

@torch.fx.wrap
def fused_conv_hardswish_flatten(bias, weight, input_tensor):
    """Wrapper function for the optimized fused kernel"""
    # Get tensor shapes  
    batch_size, input_channels, input_height, input_width = input_tensor.shape
    output_channels = bias.shape[0]
    
    # Check if input is 1x1 spatial dimensions (optimization for this specific pattern)
    assert input_height == 1 and input_width == 1, f"Expected 1x1 input, got {input_height}x{input_width}"
    
    # Create output tensor [batch_size, output_channels] - flattened from [batch, output_channels, 1, 1]
    output = torch.empty((batch_size, output_channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimize grid launch for better GPU occupancy based on batch size
    if batch_size == 1:
        # For single batch, use channels per program approach for better occupancy
        channels_per_program = 4
        total_programs = (output_channels + channels_per_program - 1) // channels_per_program
        grid_size = (total_programs, 1)
    else:
        # For larger batches, use one program per output channel per batch
        grid_size = (output_channels, batch_size)
    
    # Launch the optimized kernel with compile-time constants
    fused_conv_hardswish_flatten_kernel[grid_size](
        output_ptr=output,
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_hardswish_flatten