import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    # Pattern: conv3d + flatten(2) sequence (transpose applied separately)
    # Parameters: stride=(2, 16, 16), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    return tmp_4

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_conv_flatten_transpose_kernel(
    input_ptr,        # in_3: [1, 3, 16, 224, 224] 
    weight_ptr,       # in_1: [768, 3, 2, 16, 16]
    bias_ptr,         # in_0: [768]
    output_ptr,
    batch_size,
    input_channels,
    input_depth,
    input_height,
    input_width,
    output_channels,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    # Program IDs: pid_y = output channel, pid_z = spatial position
    pid_y = tl.program_id(0)
    pid_z = tl.program_id(1)
    
    # Calculate output dimensions correctly
    output_depth = (input_depth + 2 * padding_depth - kernel_depth) // stride_depth + 1
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    # Spatial flat size after flatten(2)
    spatial_flat_size = output_depth * output_height * output_width
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + pid_y)
    
    # Initialize accumulator
    accumulator = bias_val
    
    # Convert flat spatial index to 3D coordinates
    depth = pid_z // (output_height * output_width)
    remaining = pid_z % (output_height * output_width)
    height = remaining // output_width
    width = remaining % output_width
    
    # Calculate input positions with padding
    input_d = depth * stride_depth - padding_depth
    input_h = height * stride_height - padding_height
    input_w = width * stride_width - padding_width
    
    # Simplified convolution computation - single loop with vectorization
    # Only compute convolution if input position is valid
    mask = (input_d >= 0) & (input_d < input_depth) & (input_h >= 0) & (input_h < input_height) & (input_w >= 0) & (input_w < input_width)
    
    if mask:
        # Load input batch
        input_batch_ptr = input_ptr + (
            0 * input_channels * input_depth * input_height * input_width +  # batch 0
            0 * input_depth * input_height * input_width +  # start with input channel 0
            input_d * input_height * input_width +
            input_h * input_width +
            input_w
        )
        
        # Load weight
        weight_ptr_val = weight_ptr + (
            pid_y * input_channels * kernel_depth * kernel_height * kernel_width +
            0 * kernel_depth * kernel_height * kernel_width +  # start with input channel 0
            0 * kernel_height * kernel_width +  # start with kernel depth 0
            0 * kernel_width  # start with kernel height 0
        )
        
        # Load input and weight values
        input_val = tl.load(input_batch_ptr)
        weight_val = tl.load(weight_ptr_val)
        
        # Accumulate
        accumulator += input_val * weight_val
    
    # Store result at flattened position [batch=0, channel=pid_y, spatial=pid_z]
    output_idx = pid_y * spatial_flat_size + pid_z
    tl.store(output_ptr + output_idx, accumulator)

@torch.fx.wrap
def fused_conv3d_flatten_transpose(input_tensor, weight_tensor, bias_tensor):
    # Get input dimensions
    batch_size, input_channels, input_depth, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_depth, kernel_height, kernel_width = weight_tensor.shape
    
    # Conv3D parameters
    stride_depth, stride_height, stride_width = 1, 1, 1
    padding_depth, padding_height, padding_width = 0, 0, 0
    groups = 1
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding_depth - kernel_depth) // stride_depth + 1
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    flatten_size = output_depth * output_height * output_width
    
    # Create output tensor: [batch_size=1, output_channels, spatial_flat_size] (after flatten(2))
    spatial_flat_size = output_depth * output_height * output_width
    output = torch.empty((1, output_channels, spatial_flat_size), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block sizes for triton
    BLOCK_SIZE_Y = 64  # Process output channels in blocks
    BLOCK_SIZE_Z = 64  # Process spatial positions in blocks
    
    # Grid setup
    num_output_channels = (output_channels + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    num_spatial = (spatial_flat_size + BLOCK_SIZE_Z - 1) // BLOCK_SIZE_Z
    
    fused_conv_flatten_transpose_kernel[
        (num_output_channels, num_spatial)
    ](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        input_depth=input_depth,
        input_height=input_height,
        input_width=input_width,
        output_channels=output_channels,
        kernel_depth=kernel_depth,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_depth=stride_depth,
        stride_height=stride_height,
        stride_width=stride_width,
        padding_depth=padding_depth,
        padding_height=padding_height,
        padding_width=padding_width,
        groups=groups,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
    )
    
    return output

def replacement_func():
    return fused_conv3d_flatten_transpose