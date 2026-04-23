import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_gelu_kernel(
    # Pointers
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    # Strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_channel_stride, weight_in_channel_stride_per_group, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    # Dimensions
    N, C_in, H_in, W_in, C_out, C_per_group, groups,
    K_H, K_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + GELU kernel.
    
    Computes: out = GELU(conv(input, weight, bias))
    
    Supports both depthwise (groups=C_out) and standard (groups=1) convolutions.
    For depthwise: each output channel processes C_per_group input channels.
    For standard: all output channels share the same input channels.
    
    GELU approximation: x * sigmoid(1.702 * x) which is very close to the true GELU
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_out_pos = tl.program_id(2)
    
    # Calculate spatial position
    out_h_idx = pid_out_pos // out_w
    out_w_idx = pid_out_pos % out_w
    
    # Calculate output channel range
    c_out_base = pid_c_out * BLOCK_SIZE
    c_out_offsets = c_out_base + tl.arange(0, BLOCK_SIZE)
    c_out_mask = c_out_offsets < C_out
    
    # Load bias for this channel block
    bias = tl.load(bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Calculate input window for this output position
    in_h_start = out_h_idx * stride_h - padding_h
    in_w_start = out_w_idx * stride_w - padding_w
    
    # For depthwise: each channel processes C_per_group input channels
    # For standard: we need to iterate over all input channels
    
    if groups == C_out:
        # Depthwise convolution: each output channel is independent
        # C_per_group = 1, each output channel has its own input channel
        c_in = c_out_offsets  # Input channel = output channel (for depthwise)
        
        for k_h_idx in range(K_H):
            for k_w_idx in range(K_W):
                # Compute input position with dilation
                in_h = in_h_start + k_h_idx * dilation_h
                in_w = in_w_start + k_w_idx * dilation_w
                
                # Check bounds
                in_h_valid = (in_h >= 0) & (in_h < H_in)
                in_w_valid = (in_w >= 0) & (in_w < W_in)
                valid_mask = in_h_valid & in_w_valid
                
                # Load input for each output channel
                in_offsets = (pid_batch * input_batch_stride +
                             c_in * input_channel_stride +
                             in_h * input_height_stride +
                             in_w * input_width_stride)
                x = tl.load(input_ptr + in_offsets, mask=c_out_mask & valid_mask, other=0.0)
                
                # Load weight: [C_out, 1, K_H, K_W]
                w_offsets = (c_out_offsets * weight_out_channel_stride +
                            k_h_idx * weight_height_stride +
                            k_w_idx * weight_width_stride)
                w = tl.load(weight_ptr + w_offsets, mask=c_out_mask, other=0.0)
                
                # Multiply accumulate
                acc += w * x
    else:
        # Standard/grouped convolution
        # C_per_group input channels per group of output channels
        total_c_in_per_group = C_in // groups
        
        for k_h_idx in range(K_H):
            for k_w_idx in range(K_W):
                for c_in_idx in range(total_c_in_per_group):
                    # Calculate actual input channel
                    # For group g: output channels [g*C_per_group:(g+1)*C_per_group)
                    # Input channels are interleaved: [0, g, 2g, ...] + c_in_idx
                    
                    # Calculate which group each output channel belongs to
                    group_idx = c_out_offsets // C_per_group
                    c_in = group_idx * total_c_in_per_group + c_in_idx
                    
                    # Compute input position with dilation
                    in_h = in_h_start + k_h_idx * dilation_h
                    in_w = in_w_start + k_w_idx * dilation_w
                    
                    # Check bounds
                    in_h_valid = (in_h >= 0) & (in_h < H_in)
                    in_w_valid = (in_w >= 0) & (in_w < W_in)
                    valid_mask = in_h_valid & in_w_valid
                    
                    # Load input
                    in_offsets = (pid_batch * input_batch_stride +
                                 c_in * input_channel_stride +
                                 in_h * input_height_stride +
                                 in_w * input_width_stride)
                    x = tl.load(input_ptr + in_offsets, mask=c_out_mask & valid_mask, other=0.0)
                    
                    # Load weight: [C_out, C_in_per_group, K_H, K_W]
                    c_in_per_group_idx = c_in_idx
                    w_offsets = (c_out_offsets * weight_in_channel_stride_per_group +
                                c_in_per_group_idx * (K_H * K_W) +  # Treat 2D kernel as flat
                                k_h_idx * weight_height_stride +
                                k_w_idx * weight_width_stride)
                    w = tl.load(weight_ptr + w_offsets, mask=c_out_mask, other=0.0)
                    
                    # Multiply accumulate
                    acc += w * x
    
    # Add bias
    acc = acc + bias
    
    # Apply GELU approximation: x * sigmoid(1.702 * x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    acc_f32 = acc.to(tl.float32)
    sigmoid_arg = 1.702 * acc_f32
    # Use exp for sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    # sigmoid(x) = exp(x) / (1 + exp(x))
    exp_neg = tl.exp(-sigmoid_arg)
    sigmoid_val = 1.0 / (1.0 + exp_neg)
    gelu_out = acc_f32 * sigmoid_val
    
    # Store output (keep as float32, torch will handle conversion if needed)
    out_offsets = (pid_batch * output_batch_stride +
                  c_out_offsets * output_channel_stride +
                  out_h_idx * output_height_stride +
                  out_w_idx * output_width_stride)
    tl.store(output_ptr + out_offsets, gelu_out, mask=c_out_mask)


def _fused_conv2d_gelu_dispatch(bias, weight, input, groups, padding_h, padding_w):
    """
    Common dispatch function for fused Conv2D + GELU.
    Supports both depthwise and standard/grouped convolutions.
    """
    N, C_in, H_in, W_in = input.shape
    C_out = bias.shape[0]
    K_H, K_W = weight.shape[2], weight.shape[3]
    
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    # Calculate output dimensions
    out_h = (H_in + 2 * padding_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    out_w = (W_in + 2 * padding_w - dilation_w * (K_W - 1) - 1) // stride_w + 1
    
    # Allocate output - use float32 for better numerical stability in kernel
    output = torch.empty((N, C_out, out_h, out_w), dtype=input.dtype, device=input.device)
    
    # Grid: (N, num_channel_blocks, spatial_points)
    BLOCK_SIZE = 128
    num_c_blocks = (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    spatial_points = out_h * out_w
    grid = (N, num_c_blocks, spatial_points)
    
    # Strides
    input_batch_stride = input.stride(0)
    input_channel_stride = input.stride(1)
    input_height_stride = input.stride(2)
    input_width_stride = input.stride(3)
    
    weight_out_channel_stride = weight.stride(0)
    weight_in_channel_stride_per_group = weight.stride(1)  # This is per input channel in group
    weight_height_stride = weight.stride(2)
    weight_width_stride = weight.stride(3)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)
    
    # Calculate channels per group
    C_per_group = C_out // groups
    
    fused_conv2d_gelu_kernel[grid](
        output, input, weight, bias,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
        weight_out_channel_stride, weight_in_channel_stride_per_group, weight_height_stride, weight_width_stride,
        output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
        N, C_in, H_in, W_in, C_out, C_per_group, groups,
        K_H, K_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        out_h, out_w,
        BLOCK_SIZE,
    )
    
    return output