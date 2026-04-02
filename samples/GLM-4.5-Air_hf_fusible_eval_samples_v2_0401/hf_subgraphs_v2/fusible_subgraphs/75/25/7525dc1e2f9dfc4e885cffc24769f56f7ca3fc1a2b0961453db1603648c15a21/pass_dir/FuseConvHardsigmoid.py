import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern to match conv2d + hardsigmoid fusion"""
    conv_out = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    hardsigmoid_out = torch.nn.functional.hardsigmoid(conv_out, False)
    return hardsigmoid_out

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """Extract arguments for the fused kernel"""
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def fused_conv_hardsigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: conv2d + hardsigmoid for 1x1 convolution"""
    
    # Calculate grid and block indices
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Each program processes a portion of the output
    total_elements = batch_size * out_channels * input_height * input_width
    elements_per_pid = (total_elements + num_pid - 1) // num_pid
    start_idx = pid * elements_per_pid
    end_idx = min(start_idx + elements_per_pid, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Calculate indices
        b = idx // (out_channels * input_height * input_width)
        c_out = (idx // (input_height * input_width)) % out_channels
        h = (idx // input_width) % input_height
        w = idx % input_width
        
        # Compute output location
        output_idx = b * out_channels * input_height * input_width + c_out * input_height * input_width + h * input_width + w
        
        # Conv2D computation (1x1)
        conv_val = bias_ptr[c_out]
        
        # For 1x1 conv, we need to compute: sum_{c_in} (input[b, c_in, h, w] * weight[c_out, c_in, 0, 0])
        for c_in in range(in_channels):
            input_idx = b * in_channels * input_height * input_width + c_in * input_height * input_width + h * input_width + w
            weight_idx = c_out * in_channels * 1 * 1 + c_in * 1 * 1 + 0 * 1 + 0
            conv_val += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
        
        # Apply hardsigmoid: hardsigmoid(x) = max(0, min(1, x * 0.2 + 0.5))
        conv_val = conv_val * 0.2 + 0.5
        conv_val = tl.maximum(conv_val, 0.0)
        conv_val = tl.minimum(conv_val, 1.0)
        
        # Store result
        tl.store(output_ptr + output_idx, conv_val)

@torch.fx.wrap
def fused_conv_hardsigmoid(input_tensor, weight_tensor, bias_tensor):
    """Fused conv2d + hardsigmoid function for 1x1 convolution"""
    
    # Get input dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Create output tensor
    output_shape = (batch_size, out_channels, input_height, input_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate optimal block size
    total_elements = batch_size * out_channels * input_height * input_width
    BLOCK_SIZE = 1024  # Can be tuned
    
    # Calculate grid size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_hardsigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused conv + hardsigmoid function"""
    return fused_conv_hardsigmoid