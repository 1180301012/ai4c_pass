import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching conv2d + flatten fusion for 1x1 convolutions"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_flatten_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    INPUT_features: tl.constexpr,
    OUTPUT_features: tl.constexpr, 
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    INPUT_FEATURES_POW2: tl.constexpr,
    OUTPUT_FEATURES_POW2: tl.constexpr,
):
    """Optimized fused conv2d + flatten kernel for 1x1 convolutions"""
    
    # For debugging: First create a simple working version
    spatial_size_total = HEIGHT * WIDTH  # Always 3072
    
    # Each program processes one spatial location for one batch item
    batch_id = tl.program_id(0)
    spatial_id = tl.program_id(1)
    
    # For debugging: Just copy input to output for one feature to validate kernel
    # This will help us isolate if the issue is in memory layout or computation
    
    # Copy input[batch_id, 0, spatial_id] to output[batch_id, 0, spatial_id]
    if spatial_id < spatial_size_total:
        input_offset = batch_id * INPUT_features * spatial_size_total + spatial_id
        output_offset = batch_id * OUTPUT_features * spatial_size_total + spatial_id
        
        input_val = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def fused_conv2d_flatten(in_0, in_1, in_2):
    """Wrapper function for the fused conv2d + flatten operation"""
    
    # Get input tensor shapes and compute parameters
    batch_size = in_2.size(0)
    input_features = in_2.size(1)
    height = in_2.size(2)
    width = in_2.size(3)
    spatial_size = height * width
    output_features = in_1.size(0)
    
    # Create output tensor with flattened spatial dimensions [batch, output_features, spatial_size]
    output_shape = (batch_size, output_features, spatial_size)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Calculate grid dimensions
    # Each program processes one spatial location for one batch item
    grid_x = batch_size
    grid_y = spatial_size
    
    # Launch kernel with 2D grid
    fused_conv2d_flatten_kernel[(grid_x, grid_y)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        n_batch=batch_size,
        INPUT_features=input_features,
        OUTPUT_features=output_features,
        HEIGHT=height,
        WIDTH=width,
        INPUT_FEATURES_POW2=256,  # Next power of 2 for 160 input features
        OUTPUT_FEATURES_POW2=32,   # Next power of 2 for 17 output features
    )
    
    # Reshape output to match original conv2d + flatten pattern [batch, output_features, height, width]
    final_output = output.reshape(batch_size, output_features, height, width)
    
    return final_output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_flatten