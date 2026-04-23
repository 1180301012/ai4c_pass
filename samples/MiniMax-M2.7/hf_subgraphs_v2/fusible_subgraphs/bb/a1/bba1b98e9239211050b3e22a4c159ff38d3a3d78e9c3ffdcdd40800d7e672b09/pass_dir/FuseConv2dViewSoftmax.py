import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_view_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_stride_in,
    channel_stride_in,
    height_stride_in,
    width_stride_in,
    weight_channel_stride,
    weight_h_stride,
    weight_w_stride,
    bias_stride,
    output_batch_stride,
    output_channel_stride,
    output_seq_stride,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Block index for output position
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Each thread block handles BLOCK_SIZE elements in the sequence
    block_start = seq_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute max value for softmax numerically stable computation
    # First pass: compute max
    max_val = float("-inf")
    
    # Load and compute conv2d + max for softmax
    for h_idx in range(height):
        for w_idx in range(width):
            input_offset = (
                batch_idx * batch_stride_in +
                channel_idx * channel_stride_in +
                h_idx * height_stride_in +
                w_idx * width_stride_in
            )
            
            # Load input value
            mask = offsets < (height * width)
            inp_offset = input_offset + h_idx * height_stride_in + w_idx * width_stride_in
            inp_val = tl.load(input_ptr + inp_offset, mask=mask, other=0.0)
            
            # Load weight
            weight_offset = channel_idx * weight_channel_stride
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=1.0)
            
            # Apply conv (1x1 kernel with bias)
            # conv2d with 1x1 is: output = input * weight + bias
            bias_val = tl.load(bias_ptr, mask=True, other=0.0)
            conv_out = inp_val * weight_val + bias_val
            
            # Track max for softmax
            max_val = tl.max(max_val, conv_out)
    
    # Second pass: compute exp and sum
    exp_sum = 0.0
    for h_idx in range(height):
        for w_idx in range(width):
            mask = offsets < (height * width)
            input_offset = (
                batch_idx * batch_stride_in +
                channel_idx * channel_stride_in
            )
            inp_offset = input_offset + h_idx * height_stride_in + w_idx * width_stride_in
            inp_val = tl.load(input_ptr + inp_offset, mask=mask, other=0.0)
            
            weight_offset = channel_idx * weight_channel_stride
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=1.0)
            bias_val = tl.load(bias_ptr, mask=True, other=0.0)
            conv_out = inp_val * weight_val + bias_val
            
            exp_out = tl.exp(conv_out - max_val)
            exp_sum += exp_out
    
    # Third pass: compute softmax output
    for h_idx in range(height):
        for w_idx in range(width):
            mask = offsets < (height * width)
            input_offset = (
                batch_idx * batch_stride_in +
                channel_idx * channel_stride_in
            )
            inp_offset = input_offset + h_idx * height_stride_in + w_idx * width_stride_in
            inp_val = tl.load(input_ptr + inp_offset, mask=mask, other=0.0)
            
            weight_offset = channel_idx * weight_channel_stride
            weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=1.0)
            bias_val = tl.load(bias_ptr, mask=True, other=0.0)
            conv_out = inp_val * weight_val + bias_val
            
            exp_out = tl.exp(conv_out - max_val)
            softmax_out = exp_out / exp_sum
            
            # Store output: (batch, 1, batch * channels * height * width)
            # After view: (batch, 1, channels * height * width)
            # With channel_idx and flattened spatial dims
            spatial_idx = h_idx * width + w_idx
            global_seq_idx = channel_idx * height * width + spatial_idx
            
            output_offset = (
                batch_idx * output_batch_stride +
                0 * output_channel_stride +
                global_seq_idx * output_seq_stride
            )
            tl.store(output_ptr + output_offset, softmax_out, mask=mask)


@torch.fx.wrap
def fused_conv2d_view_softmax_wrapper(input_tensor, weight_tensor, bias_tensor, batch, channels, height, width):
    """
    Fused kernel for: conv2d + view + softmax
    input_tensor: (batch, channels, height, width)
    weight_tensor: (1, channels, 1, 1)
    bias_tensor: (1,)
    output: (batch, 1, channels * height * width)
    """
    # Prepare output tensor
    output_shape = (batch, 1, channels * height * width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Compute strides
    batch_stride_in = channels * height * width
    channel_stride_in = height * width
    height_stride_in = width
    width_stride_in = 1
    
    weight_channel_stride = 1  # weight shape (1, channels, 1, 1)
    weight_h_stride = 1
    weight_w_stride = 1
    
    bias_stride = 1
    
    output_batch_stride = channels * height * width
    output_channel_stride = channels * height * width
    output_seq_stride = 1
    
    # Grid configuration
    BLOCK_SIZE = 64
    
    grid = (batch, channels, (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    fused_conv2d_view_softmax_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output,
        batch_stride_in,
        channel_stride_in,
        height_stride_in,
        width_stride_in,
        weight_channel_stride,
        weight_h_stride,
        weight_w_stride,
        bias_stride,
        output_batch_stride,
        output_channel_stride,
        output_seq_stride,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d + view + softmax
    in_0: bias tensor (shape: [1])
    in_1: weight tensor (shape: [1, 512, 1, 1])
    in_2: input tensor (shape: [batch, 512, height, width])
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(tmp_2.shape[0], 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2)


def replacement_func():
    """Return the fused kernel wrapper"""
    return fused_conv2d_view_softmax_wrapper