import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Match Conv2D → Hardswish (inplace) → Flatten pattern"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """Extract arguments for the replacement kernel"""
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def fused_conv_hardswish_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Conv2D + Hardswish + Flatten operations"""
    pid = tl.program_id(0)
    
    # Each thread handles one output element
    if pid >= batch_size * out_channels:
        return
    
    # Decompose program ID into batch and output channel indices
    batch_idx = pid // out_channels
    out_channel_idx = pid % out_channels
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + out_channel_idx)
    
    # Initialize the sum with the bias
    result = bias
    
    # Specialize for 1x1 conv - optimize inner loop for performance
    # For 1x1 conv with stride=1, padding=0: output_ch = sum(in_ch, weight[out_ch][in_ch] * input[batch][in_ch])
    for in_ch in range(in_channels):
        # Optimized memory access pattern for 1x1 conv
        weight_ptr_offset = out_channel_idx * in_channels + in_ch
        input_ptr_offset = batch_idx * in_channels + in_ch
        
        # Load weight and input values
        weight_val = tl.load(weight_ptr + weight_ptr_offset)
        input_val = tl.load(input_ptr + input_ptr_offset)
        
        # Fused multiply-add
        result += weight_val * input_val
    
    # Apply hardswish activation: x * relu6(x + 3) / 6
    x = result
    relu6 = tl.maximum(0.0, tl.minimum(6.0, x + 3.0))
    hardswish_val = x * relu6 * 0.16666666666666666  # Division by 6 optimized
    
    # Store the result in the flat output tensor
    tl.store(output_ptr + pid, hardswish_val)

@torch.fx.wrap
def fused_conv_hardswish_flatten(input_tensor, weight_tensor, bias_tensor):
    """
    Fused Conv2D + Hardswish + Flatten operation
    
    Args:
        input_tensor: [batch, in_channels, height, width]
        weight_tensor: [out_channels, in_channels, kernel_h, kernel_w] 
        bias_tensor: [out_channels]
    
    Returns:
        Flattened output of shape [batch * out_channels]
    """
    # Get tensor shapes
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, weight_height, weight_width = weight_tensor.shape
    
    # Calculate output dimensions for conv2d
    # output_height = floor((input_height + 2*padding - dilation*(kernel_height-1) - 1)/stride + 1)
    # output_width = floor((input_width + 2*padding - dilation*(kernel_width-1) - 1)/stride + 1)
    # For our case: stride=1, padding=0, dilation=1, kernel_size=1x1, input_size=1x1
    # Both output dimensions = floor((1 + 0 - 1*(1-1) - 1)/1 + 1) = floor(0 + 1) = 1
    output_height = ((input_height - weight_height + 2 * 0) // 1) + 1
    output_width = ((input_width - weight_width + 2 * 0) // 1) + 1
    
    # The expected output size after flatten: batch_size * out_channels * output_height * output_width
    output_size = batch_size * out_channels * output_height * output_width
    
    # Allocate output tensor
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Determine optimal block size based on output size for better GPU occupancy
    if output_size <= 4096:
        BLOCK_SIZE = 256
    elif output_size <= 16384:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size: one program per output element
    grid_size = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel - use grid size calculated correctly
    fused_conv_hardswish_flatten_kernel[(grid_size,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the fused kernel function"""
    return fused_conv_hardswish_flatten