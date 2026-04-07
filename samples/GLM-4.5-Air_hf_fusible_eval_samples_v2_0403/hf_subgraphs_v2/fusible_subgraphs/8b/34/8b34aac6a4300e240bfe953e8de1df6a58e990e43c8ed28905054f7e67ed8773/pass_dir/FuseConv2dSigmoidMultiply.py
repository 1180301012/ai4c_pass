import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, multiply_input):
    """
    Pattern: Conv2D (1x1) → Sigmoid → Multiply with exact signature matching
    This optimizes the second computational branch: in_5 → conv2d → sigmoid → multiply
    """
    # 1x1 convolution with fused sigmoid and multiplication
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
    sigmoid_result = torch.sigmoid(conv_result)
    final_result = multiply_input * sigmoid_result
    return final_result

def replacement_args(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, multiply_input):
    return (input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, multiply_input)

@triton.jit
def fused_conv_sigmoid_multiply_kernel(
    input_ptr, weight_ptr, bias_ptr, multiply_ptr, output_ptr,
    batch_size, in_channels, out_channels, 
    input_height, input_width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for fused convolution + sigmoid + multiply
    Uses 1x1 convolution which is essentially a pointwise linear transformation
    """
    pid = tl.program_id(0)
    
    # Each program handles one channel in the output
    if pid >= out_channels:
        return
    
    # Load bias for this channel
    bias = tl.load(bias_ptr + pid)
    
    # Calculate output spatial dimensions (remain same due to 1x1 conv, stride 1, padding 0)
    output_height = input_height
    output_width = input_width
    
    # Process all spatial locations for this channel
    for h in range(0, output_height, BLOCK_SIZE):
        for w in range(0, output_width, BLOCK_SIZE):
            # Calculate spatial offsets
            h_idx = h + tl.arange(0, BLOCK_SIZE)
            w_idx = w + tl.arange(0, BLOCK_SIZE)
            
            # Create spatial mask
            h_mask = h_idx < output_height
            w_mask = w_idx < output_width
            spatial_mask = h_mask[:, None] & w_mask[None, :]
            
            # Flatten spatial indices for efficient loading
            h_flat = h_idx[:, None].repeat_interleave(output_width, dim=1)
            w_flat = w_idx[None, :].repeat(output_height, 1)
            flat_indices = h_flat * output_width + w_flat
            flat_mask = spatial_mask.flatten()
            
            # Load 1x1 weights (single weight per output channel)
            weight = tl.load(weight_ptr + pid * in_channels + tl.arange(0, in_channels), mask=tl.arange(0, in_channels) < in_channels)
            
            # Load multiply input values for this channel
            multiply_input_ptrs = multiply_ptr + pid * (output_height * output_width) + flat_indices
            multiply_vals = tl.load(multiply_input_ptrs, mask=flat_mask[:, None], other=0.0)
            
            # Compute convolution (1x1 is just weighted sum + bias)
            conv_val = bias
            for c in range(in_channels):
                if c < in_channels:
                    # Load input values for channel c
                    input_ptrs = input_ptr + c * (batch_size * input_height * input_width) + pid * (output_height * output_width) + flat_indices
                    input_vals = tl.load(input_ptrs, mask=flat_mask[:, None], other=0.0)
                    conv_val += weight[c] * input_vals
            
            # Apply sigmoid
            sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
            
            # Apply multiplication
            output_val = multiply_vals * sigmoid_val
            
            # Store result
            output_ptrs = output_ptr + pid * (output_height * output_width) + flat_indices
            tl.store(output_ptrs, output_val, mask=flat_mask[:, None])

@torch.fx.wrap
def fused_conv_sigmoid_multiply(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups, multiply_input):
    """
    Wrapper function for the fused conv2d + sigmoid + multiply operation
    """
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Create output tensor with same shape as input (1x1 conv preserves spatial dims)
    output_shape = (batch_size, out_channels, input_height, input_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set block size for vectorized memory access
    BLOCK_SIZE = 16  # Optimized for GPU occupancy
    
    # Calculate grid size (one program per output channel)
    grid_size = out_channels
    
    # Launch the Triton kernel
    fused_conv_sigmoid_multiply_kernel[grid_size](
        input_tensor,
        weight_tensor, 
        bias_tensor,
        multiply_input,
        output,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return fused_conv_sigmoid_multiply