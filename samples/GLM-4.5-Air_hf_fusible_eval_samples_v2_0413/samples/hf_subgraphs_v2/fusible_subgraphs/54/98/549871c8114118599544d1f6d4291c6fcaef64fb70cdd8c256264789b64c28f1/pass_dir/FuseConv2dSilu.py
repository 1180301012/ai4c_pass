import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias):
    """Pattern: Conv2D followed by SILU activation"""
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    silu_out = torch.nn.functional.silu(conv_out, inplace=False)
    # Return all observable outputs that match the model's original structure
    return silu_out

def replacement_args(conv_input, conv_weight, conv_bias):
    """Extract arguments for the fused kernel"""
    return (conv_input, conv_weight, conv_bias)

@triton.jit
def fused_conv2d_silu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + SILU kernel"""
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Each thread block handles a specific output location across all batches and output channels
    out_c = pid // (out_height * out_width)
    out_h = (pid % (out_height * out_width)) // out_width
    out_w = pid % out_width
    
    # Skip if out of bounds
    if out_c >= out_channels or out_h >= out_height or out_w >= out_width:
        return
    
    # Calculate input coordinates for convolution
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w
    
    # Compute convolution result
    val = 0.0
    if 0 <= in_h < in_height and 0 <= in_w < in_width:
        for oc in range(out_channels):
            bias_val = tl.load(bias_ptr + oc)
            for ic in range(in_channels):
                # Load weight
                weight_idx = oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + 0 * kernel_w + 0
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Load input
                input_idx = oc * in_height * in_width + in_h * in_width + in_w
                input_val = tl.load(input_ptr + input_idx)
                
                val += weight_val * input_val
            val += bias_val
    
    # Apply SILU activation: x * sigmoid(x)
    sigmoid_val = 1.0 / (1.0 + torch.exp(-val))
    output_val = val * sigmoid_val
    
    # Store result
    out_idx = out_c * out_height * out_width + out_h * out_width + out_w
    tl.store(output_ptr + out_idx, output_val)

@torch.fx.wrap
def fused_conv2d_silu(conv_input, conv_weight, conv_bias):
    """Wrapper function for the fused Conv2D + SILU operation"""
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = conv_input.shape
    out_channels, _, kernel_h, kernel_w = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * 0 - kernel_h) // 1 + 1  # stride=(1,1), padding=(0,0)
    out_width = (in_width + 2 * 0 - kernel_w) // 1 + 1
    
    # Calculate total number of output elements
    total_elements = batch_size * out_channels * out_height * out_width
    
    # Determine block size and grid size
    BLOCK_SIZE = 256
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (using float32 for precision, will be cast back to original dtype)
    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=torch.float32)
    
    # Launch kernel
    fused_conv2d_silu_kernel[grid_size](
        conv_input,
        conv_weight,
        conv_bias,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        1, 1,  # stride_h, stride_w
        0, 0,  # pad_h, pad_w
        BLOCK_SIZE
    )
    
    # Cast output back to original dtype (to match original behavior)
    return output.to(conv_input.dtype)

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv2d_silu