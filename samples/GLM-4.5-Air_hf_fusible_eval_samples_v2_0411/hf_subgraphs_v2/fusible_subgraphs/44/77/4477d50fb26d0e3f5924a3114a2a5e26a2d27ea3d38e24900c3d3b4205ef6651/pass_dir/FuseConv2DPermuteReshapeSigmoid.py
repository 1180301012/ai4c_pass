import torch
import triton
import triton.language as tl
from typing import Tuple

# Pattern matching function - matches the exact computation pattern
def pattern(arg0, arg1, arg2):
    """
    Matches: conv2d -> permute -> reshape -> sigmoid pattern
    This pattern appears consistently across all graph variations.
    """
    # Exact sequence from the original computation
    conv2d = torch.conv2d(arg2, arg1, arg0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(conv2d.shape[0], -1, arg0.shape[0])
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5

def replacement_args(arg0, arg1, arg2):
    """
    Extract arguments needed for the replacement kernel.
    Returns a tuple of all necessary tensor arguments.
    """
    return (arg0, arg1, arg2)

@triton.jit
def fused_conv_sigmoid_kernel(
    bias_ptr,           # Bias tensor: [B]
    weight_ptr,         # Weight tensor: [B, C_in, 1, 1] - flattened
    input_ptr,          # Input tensor: [N, C_in, H, W]
    output_ptr,         # Output tensor: [N, H*W, B]
    n_batch: tl.constexpr,
    n_input_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. 1x1 convolution with bias
    2. NHWC format conversion 
    3. Reshape to [N, H*W, B]
    4. Sigmoid activation
    
    All operations fused to avoid intermediate memory allocations.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    # Each thread computes one output element: output[batch, spatial_position, output_channel]
    # Output shape: [N, H*W, B]
    
    # Decompose offset to batch, spatial position, and output channel
    spatial_elements_per_batch = height * width
    elements_per_batch_output = spatial_elements_per_batch * n_batch
    
    batch_idx = offset // elements_per_batch_output
    remaining_offset = offset % elements_per_batch_output
    
    output_channel_idx = remaining_offset // spatial_elements_per_batch
    spatial_flat_idx = remaining_offset % spatial_elements_per_batch
    
    # Convert spatial flat index to h, w coordinates
    h_idx = spatial_flat_idx // width
    w_idx = spatial_flat_idx % width
    
    # Compute 1x1 convolution: sum over input channels
    conv_sum = 0.0
    for c_in_idx in range(n_input_channels):
        # Load input at [batch_idx, c_in_idx, h_idx, w_idx]
        input_offset = batch_idx * (n_input_channels * height * width) + \
                       c_in_idx * (height * width) + h_idx * width + w_idx
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Load weight for [output_channel_idx, c_in_idx, 0, 0]
        weight_offset = output_channel_idx * (n_input_channels * 1 * 1) + \
                       c_in_idx * (1 * 1) + 0
        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        conv_sum += input_val * weight_val
    
    # Add bias for this output channel
    bias_val = tl.load(bias_ptr + output_channel_idx, mask=mask, other=0.0)
    conv_output = conv_sum + bias_val
    
    # Apply sigmoid activation using Triton operations
    sigmoid_output = 1.0 / (1.0 + tl.exp(-conv_output))
    
    # Store result at the computed offset
    tl.store(output_ptr + offset, sigmoid_output, mask=mask)

@torch.fx.wrap 
def fused_conv_sigmoid_forward(bias, weight, input_tensor):
    """
    Fused forward pass that combines conv2d + permute + reshape + sigmoid.
    
    Args:
        bias: [B] - bias tensor
        weight: [B, C_in, 1, 1] - weight tensor  
        input_tensor: [N, C_in, H, W] - input tensor
        
    Returns:
        output: [N, H*W, B] - sigmoid activated outputs
    """
    # Get tensor shapes and properties
    B = bias.shape[0]  # Number of output channels
    C_in = weight.shape[1]  # Number of input channels
    N = input_tensor.shape[0]  # Batch size
    H = input_tensor.shape[2]  # Height
    W = input_tensor.shape[3]  # Width
    
    # Total number of spatial elements per batch: H * W
    total_spatial_elements = H * W
    
    # Total elements in final output [N, H*W, B]
    total_output_elements = N * total_spatial_elements * B
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_output_elements, BLOCK_SIZE),)
    
    # Create output tensor: [N, H*W, B]
    output = torch.empty((N, H * W, B), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the fused kernel that does everything
    fused_conv_sigmoid_kernel[grid_size](
        bias_ptr=bias,
        weight_ptr=weight.view(-1),  # Flatten for easier indexing
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=B,
        n_input_channels=C_in, 
        height=H,
        width=W,
        total_elements=total_output_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """
    Returns the fused function as a zero-argument function.
    This allows the framework to call it when the pattern is matched.
    """
    return fused_conv_sigmoid_forward