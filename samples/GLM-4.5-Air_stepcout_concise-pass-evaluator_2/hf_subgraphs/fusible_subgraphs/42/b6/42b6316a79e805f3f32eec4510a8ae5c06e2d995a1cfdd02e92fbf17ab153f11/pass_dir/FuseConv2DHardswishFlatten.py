import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Conv2D operation with in-place hardswish followed by flatten
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    # Extract arguments: bias, weight, input
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_hardswish_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr,
    batch_size, out_channels, in_channels,
    BLOCK_SIZE_OUT: tl.constexpr
):
    # Compute program IDs
    pid_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Calculate output range for this program
    out_range = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    
    # Initialize accumulator array for this program
    accumulator = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    
    # Perform matrix multiplication: bias + sum(weight_input)
    for k in range(0, in_channels):
        # Load weight for current output channel and input channel
        # Shape: [BLOCK_SIZE_OUT]
        weight = tl.load(weight_ptr + out_range * in_channels + k,
                        mask=out_range < out_channels,
                        other=0.0)
        
        # Load input for current batch and input channel
        # Shape: scalar
        input_val = tl.load(input_ptr + pid_b * in_channels + k,
                           mask=k < in_channels,
                           other=0.0)
        
        # Accumulate the product: weight * input_val
        accumulator += weight * input_val
    
    # Add bias
    # Shape: [BLOCK_SIZE_OUT]
    bias = tl.load(bias_ptr + out_range,
                   mask=out_range < out_channels,
                   other=0.0)
    
    # Apply hardswish: x * relu6(x + 3) / 6
    # Shape: [BLOCK_SIZE_OUT]
    temp = accumulator + bias + 3
    relu6_result = tl.maximum(tl.minimum(temp, 6), 0)
    hardswish_result = (accumulator + bias) * relu6_result / 6
    
    # Store result to 2D tensor [batch_size, out_channels]
    # Calculate row-major indices for 2D tensor
    batch_indices = pid_b
    channel_indices = out_range
    
    # Create flattened indices for row-major storage
    output_indices = batch_indices * out_channels + channel_indices
    tl.store(output_ptr + output_indices, hardswish_result, mask=(channel_indices < out_channels) & (batch_indices < batch_size))

@torch.fx.wrap
def fused_conv_hardswish(bias, weight, input_tensor):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    
    # For 1x1 convolution with 1x1 spatial dims, 
    # Output should be [batch_size, out_channels] - same as flatten(1, -1) result
    output_shape = (batch_size, out_channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set block sizes for better GPU occupancy
    BLOCK_SIZE_OUT = 128  # Number of output channels to process per program
    
    # Calculate grid dimensions
    num_blocks_out = (out_channels + BLOCK_SIZE_OUT - 1) // BLOCK_SIZE_OUT
    num_blocks_batch = batch_size
    
    # Launch kernel - use 1D output tensor and reshape if needed
    fused_conv_hardswish_kernel[(num_blocks_out, num_blocks_batch)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        out_channels=out_channels,
        in_channels=in_channels,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
    )
    
    return output

def replacement_func():
    return fused_conv_hardswish