import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """
    Pattern matches Conv2D + Padding sequence
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    return tmp_2

def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement kernel
    """
    return (in_0, in_1)

@triton.jit
def fused_conv2d_padding_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch,
    in_channels,
    out_channels,
    in_height,
    in_width,
    padding,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused kernel for Conv2D + padding operations
    """
    # Output dimensions after padding
    out_height_padded = in_height + 2 * padding
    out_width_padded = in_width + 2 * padding
    
    # Program ID for batch and spatial coordinates
    pid = tl.program_id(0)
    batch_id = pid // (out_height_padded * out_width_padded)
    spatial_pid = pid % (out_height_padded * out_width_padded)
    out_h = spatial_pid // out_width_padded
    out_w = spatial_pid % out_width_padded
    
    # If we're in the padded region, output zeros
    if (out_h < padding or out_h >= in_height + padding or 
        out_w < padding or out_w >= in_width + padding):
        # Output zero for padding region
        for c_out in range(0, out_channels, BLOCK_SIZE_N):
            channels_block = min(BLOCK_SIZE_N, out_channels - c_out)
            out_offset = ((batch_id * out_height_padded + out_h) * out_width_padded + out_w) * out_channels + c_out
            for c in range(channels_block):
                tl.store(output_ptr + out_offset + c, 0.0)
        return
    
    # Convert to input coordinates (remove padding)
    in_h = out_h - padding
    in_w = out_w - padding
    
    # For 1x1 convolution, we only need to sum over input channels for each position
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, in_channels, BLOCK_SIZE_K):
        channels_block = min(BLOCK_SIZE_K, in_channels - k)
        
        # Load input element
        input_offset = (batch_id * in_height + in_h) * in_width + in_w
        input_ptrs = input_ptr + input_offset + k
        input_vals = tl.load(input_ptrs, mask=k < in_channels, other=0.0).to(tl.float32)
        
        # Load weight for output channels
        weight_ptrs = weight_ptr + k * out_channels
        weight_vals = tl.load(weight_ptrs, mask=tl.arange(0, BLOCK_SIZE_N) < out_channels, other=0.0)
        
        # Matrix multiplication (input channels x output channels)
        for c_out in range(BLOCK_SIZE_N):
            if c_out < out_channels:
                acc[c_out] += tl.sum(input_vals * weight_vals[c_out])
    
    # Store result
    for c_out in range(0, out_channels, BLOCK_SIZE_N):
        channels_block = min(BLOCK_SIZE_N, out_channels - c_out)
        if channels_block > 0:
            out_offset = ((batch_id * out_height_padded + out_h) * out_width_padded + out_w) * out_channels + c_out
            tl.store(output_ptr + out_offset + acc[c_out:c_out+channels_block], c_out < out_channels)

@torch.fx.wrap
def fused_conv2d_padding_func(input_tensor, weight_tensor, padding=2):
    """
    Wrapper function for the fused Conv2D + padding kernel
    """
    # Get input tensor properties
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    
    # Get shapes
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    
    # For 1x1 convolution, kernel should be 1x1
    assert kernel_h == 1 and kernel_w == 1, "Only 1x1 convolution supported"
    
    # Calculate output shape including padding
    out_height_padded = in_height + 2 * padding
    out_width_padded = in_width + 2 * padding
    output_shape = (batch, out_channels, out_height_padded, out_width_padded)
    
    # Create output tensor
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 256  # Batch * spatial work per program
    BLOCK_SIZE_N = 32   # Output channels per program
    BLOCK_SIZE_K = 32   # Input channels per program
    
    # Calculate grid size
    spatial_elements = out_height_padded * out_width_padded
    total_elements = batch * spatial_elements
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE_M),)
    
    # Launch kernel
    fused_conv2d_padding_kernel[grid_size](
        input_tensor,
        weight_tensor,
        output_tensor,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        padding,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output_tensor

def replacement_func():
    """
    Returns the fused function as a zero-argument function
    """
    return lambda in_0, in_1: fused_conv2d_padding_func(in_1, in_0, 2)