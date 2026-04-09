import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and channel parallelization
    batch_pid = tl.program_id(0)
    channel_pid = tl.program_id(1)
    
    # Calculate thread indices
    batch_offset = batch_pid
    channel_offset = channel_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_offset < out_channels
    
    # For each output channel, compute conv2d result
    conv_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Standard conv2d calculation for 1x1 convolution
    # For output channel c_out: sum_{c_in} (input[b][c_in][0][0] * weight[c_out][c_in][0][0]) + bias[c_out]
    for ic in range(in_channels):
        # Load input value for this input channel
        input_offset = input_ptr + (batch_offset * in_channels + ic) * 1 * 1  # [1,1] spatial
        input_val = tl.load(input_offset, mask=True, other=0.0)
        
        # Load weight values for this input channel across all output channels
        weight_offset = weight_ptr + (channel_offset * in_channels + ic) * 1 * 1
        weight_vals = tl.load(weight_offset, mask=channel_mask, other=0.0)
        
        # Add weighted input to result
        conv_result += input_val * weight_vals
        
        # Add bias for each output channel
    bias_offset = bias_ptr + channel_offset
    bias_vals = tl.load(bias_offset, mask=channel_mask, other=0.0)
    conv_result += bias_vals
    
    # Apply sigmoid activation
    sigmoid_result = tl.sigmoid(conv_result)
    
    # Store result - output has shape [batch, out_channels, 1, 1]
    output_offset = output_ptr + (batch_offset * out_channels + channel_offset) * 1 * 1
    tl.store(output_offset, sigmoid_result, mask=channel_mask)

def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@torch.fx.wrap  
def fused_conv_sigmoid_forward(in_3, in_1, in_0):
    """Fused conv2d + sigmoid operation for 1x1 convolution"""
    batch_size, in_channels, height, width = in_3.shape
    out_channels = in_1.shape[0]
    
    # Grid setup: [batch, out_channels] dimensions (no spatial dimension since 1x1 conv)
    BLOCK_SIZE = 256  # Number of channels per program
    batch_grid = batch_size
    channel_grid = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with shape [batch, out_channels, 1, 1]
    output = torch.empty((batch_size, out_channels, 1, 1), dtype=in_1.dtype, device=in_3.device)
    
    # Launch fused kernel
    fused_conv_sigmoid_kernel[(batch_grid, channel_grid)](
        in_3,
        in_1,
        in_0,
        output,
        batch_size,
        in_channels,
        out_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv_sigmoid_forward