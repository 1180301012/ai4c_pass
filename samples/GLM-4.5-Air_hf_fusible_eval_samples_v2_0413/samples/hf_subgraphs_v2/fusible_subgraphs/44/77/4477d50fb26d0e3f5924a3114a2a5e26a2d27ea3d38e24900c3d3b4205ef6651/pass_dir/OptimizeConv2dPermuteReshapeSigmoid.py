import torch
import triton
import triton.language as tl
import math

def pattern(conv2d_input, conv2d_weight, conv2d_bias):
    """
    Pattern matching for Conv2D + Permute operations.
    This matches the core computation that can be optimized.
    """
    conv2d_result = torch.conv2d(conv2d_input, conv2d_weight, conv2d_bias, (1, 1), (0, 0), (1, 1), 1)
    permuted_result = conv2d_result.permute(0, 2, 3, 1)
    return permuted_result


def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """
    Extract arguments needed for the optimization.
    """
    return (input_tensor, weight_tensor, bias_tensor)


@triton.jit
def optimized_conv2d_permute_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    weight_channels_in,
    kernel_h,
    kernel_w,
    block_size: tl.constexpr,
):
    """
    Optimized kernel for Conv2D + Permute operations.
    Uses vectorized memory access with proper correctness guarantees.
    """
    pid = tl.program_id(0)
    total_elements = input_batch * input_height * input_width * output_channels
    
    # Each thread handles multiple consecutive elements for better memory coalescing
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Convert offset to coordinates in the permuted layout [batch, height, width, channels]
    batch_idx = offsets // (input_height * input_width * output_channels)
    spatial_idx = (offsets // output_channels) % (input_height * input_width)
    channel_idx = offsets % output_channels
    
    # Convert spatial index to height/width coordinates
    h_idx = spatial_idx // input_width
    w_idx = spatial_idx % input_width
    
    # For each element, compute the 1x1 convolution
    results = tl.zeros((block_size,), dtype=tl.float32)
    
    for ci in range(input_channels):
        # Calculate source offsets for input tensor [batch, channels, height, width]
        input_offsets = (batch_idx * input_channels + ci) * input_height * input_width + h_idx * input_width + w_idx
        
        # Load input values with vectorized access
        input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Calculate weight offsets [output_channels, input_channels]
        weight_offsets = channel_idx * input_channels + ci
        
        # Load weight values
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
        
        # Compute convolution result: sum(input * weight)
        results += input_vals * weight_vals
    
    # Add bias after the convolution (only once)
    bias_vals = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    results += bias_vals
    
    # Store results in the permuted layout [batch, height, width, channels]
    tl.store(output_ptr + offsets, results, mask=mask)


@torch.fx.wrap  
def optimized_conv2d_permute_op(input_tensor, weight_tensor, bias_tensor):
    """
    Fused operation wrapper that launches the optimized Triton kernel.
    Handles the Conv2D + Permute operations for better performance.
    """
    # Get input tensor dimensions
    input_shape = input_tensor.shape
    weight_shape = weight_tensor.shape
    
    # Extract dimensions
    input_batch, input_channels, input_height, input_width = input_shape
    output_channels, weight_channels_in, kernel_h, kernel_w = weight_shape
    
    # Output of conv2d + permute will be [batch, height, width, channels]
    output_elements = input_batch * input_height * input_width * output_channels
    
    # Create output tensor in permuted layout [batch, height, width, channels]
    output_tensor = torch.empty([input_batch, input_height, input_width, output_channels], 
                                dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimized kernel launch parameters for better GPU utilization
    # Adjust block size based on tensor characteristics
    if output_elements > 1000000:
        block_size = 256  # Large tensors benefit from larger blocks
    elif output_elements > 100000:
        block_size = 128  # Medium tensors
    else:
        block_size = 64   # Small tensors
    
    num_blocks = (output_elements + block_size - 1) // block_size
    
    # Launch optimized kernel
    optimized_conv2d_permute_kernel[(num_blocks,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output_tensor,
        input_batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_channels_in,
        kernel_h,
        kernel_w,
        block_size
    )
    
    return output_tensor


def replacement_func():
    """
    Returns the optimized fused function.
    """
    return optimized_conv2d_permute_op