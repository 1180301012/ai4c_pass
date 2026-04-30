"""
Fused kernel for Conv2D + Add + Flatten + Transpose + LayerNorm + Transpose pattern.

This pass fuses multiple operations into a single Triton kernel for maximum performance:
1. Conv2D with bias
2. Element-wise add with residual connection
3. Flatten spatial dimensions (16x16 -> 256)
4. Transpose for layer normalization
5. LayerNorm (mean=0, variance=1, normalize, scale, shift)
6. Transpose output (twice, for return values)

The optimization achieves:
- Single kernel launch instead of 6+ operations
- No intermediate tensor allocations
- Optimal memory coalescing and GPU utilization
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_add_layernorm_kernel(
    # Input pointers
    input_ptr, weight_ptr, bias_ptr, residual_ptr,
    # LayerNorm parameters
    ln_weight_ptr, ln_bias_ptr,
    # Output pointers
    out_transposed_ptr, out_ptr,
    # Strides for input
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    # Strides for weight
    weight_out_channel_stride, weight_in_channel_stride, weight_kernel_h_stride, weight_kernel_w_stride,
    # Strides for residual
    residual_batch_stride, residual_channel_stride, residual_h_stride, residual_w_stride,
    # Output strides
    out_batch_stride, out_seq_stride, out_channel_stride,
    # Dimensions
    batch_size, in_channels, out_channels, height, width,
    kernel_size, padding, stride,
    # LayerNorm parameters
    ln_num_elements, eps: tl.constexpr,
    # Block size for layer norm
    LN_BLOCK_SIZE: tl.constexpr,
    # For autotuning
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Depthwise Conv2D with bias
    2. Add residual connection
    3. Reshape to (batch, seq_len, channels)
    4. LayerNorm
    5. Return transposed and non-transposed outputs
    """
    # Get position indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate output position (each thread handles one output channel)
    output_offset = (
        pid_batch * out_batch_stride +
        (pid_h * width + pid_w) * out_seq_stride
    )
    
    # Compute mean and variance for layer norm
    # Each thread loads all channels for its position and computes stats
    sum_val = 0.0
    sq_sum_val = 0.0
    
    # Process all output channels
    for c in range(out_channels):
        # Calculate conv offset for this output channel
        weight_offset = c * weight_out_channel_stride
        
        # Depthwise conv: each output channel corresponds to one input channel
        conv_result = 0.0
        
        # Load bias
        bias_val = tl.load(bias_ptr + c).to(tl.float32)
        conv_result = bias_val
        
        # Load weight and do depthwise convolution
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = pid_h * stride + kh - padding
                iw = pid_w * stride + kw - padding
                
                # Check bounds
                if 0 <= ih < height and 0 <= iw < width:
                    # Input load offset
                    input_offset = (
                        pid_batch * input_batch_stride +
                        c * input_channel_stride +
                        ih * input_h_stride +
                        iw * input_w_stride
                    )
                    # Weight load offset
                    w_offset = weight_offset + kh * weight_kernel_h_stride + kw * weight_kernel_w_stride
                    
                    input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                    weight_val = tl.load(weight_ptr + w_offset).to(tl.float32)
                    conv_result += input_val * weight_val
        
        # Load residual and add
        residual_offset = (
            pid_batch * residual_batch_stride +
            c * residual_channel_stride +
            pid_h * residual_h_stride +
            pid_w * residual_w_stride
        )
        residual_val = tl.load(residual_ptr + residual_offset).to(tl.float32)
        conv_result += residual_val
        
        sum_val += conv_result
        sq_sum_val += conv_result * conv_result
    
    # Compute mean and variance
    mean = sum_val / tl.cast(ln_num_elements, tl.float32)
    var = (sq_sum_val / tl.cast(ln_num_elements, tl.float32)) - mean * mean
    std = tl.sqrt(var + eps)
    
    # Normalize and store for each output channel
    for c in range(out_channels):
        # Calculate conv offset for this output channel
        weight_offset = c * weight_out_channel_stride
        
        # Depthwise conv: each output channel corresponds to one input channel
        conv_result = 0.0
        
        # Load bias
        bias_val = tl.load(bias_ptr + c).to(tl.float32)
        conv_result = bias_val
        
        # Load weight and do depthwise convolution
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = pid_h * stride + kh - padding
                iw = pid_w * stride + kw - padding
                
                # Check bounds
                if 0 <= ih < height and 0 <= iw < width:
                    # Input load offset
                    input_offset = (
                        pid_batch * input_batch_stride +
                        c * input_channel_stride +
                        ih * input_h_stride +
                        iw * input_w_stride
                    )
                    # Weight load offset
                    w_offset = weight_offset + kh * weight_kernel_h_stride + kw * weight_kernel_w_stride
                    
                    input_val = tl.load(input_ptr + input_offset).to(tl.float32)
                    weight_val = tl.load(weight_ptr + w_offset).to(tl.float32)
                    conv_result += input_val * weight_val
        
        # Load residual and add
        residual_offset = (
            pid_batch * residual_batch_stride +
            c * residual_channel_stride +
            pid_h * residual_h_stride +
            pid_w * residual_w_stride
        )
        residual_val = tl.load(residual_ptr + residual_offset).to(tl.float32)
        conv_result += residual_val
        
        # LayerNorm
        ln_weight = tl.load(ln_weight_ptr + c).to(tl.float32)
        ln_bias_val = tl.load(ln_bias_ptr + c).to(tl.float32)
        normalized = (conv_result - mean) / std
        output_val = normalized * ln_weight + ln_bias_val
        
        # Store transposed output (channels first dimension at position 0)
        tl.store(out_transposed_ptr + output_offset + c * out_channel_stride, output_val)
        
        # Store non-transposed output (seq_len first dimension at position 0)
        out_offset = output_offset + c * out_channel_stride
        tl.store(out_ptr + out_offset, output_val)


@torch.fx.wrap
def fused_conv_add_layernorm_wrapper(
    input_tensor, weight, bias, residual,
    ln_weight, ln_bias, eps=1e-05
):
    """
    Wrapper function that launches the fused kernel.
    
    Args:
        input_tensor: Input tensor [batch, channels, height, width]
        weight: Conv2D weight [out_channels, 1, 3, 3] (depthwise)
        bias: Conv2D bias [out_channels]
        residual: Residual connection tensor (same as input)
        ln_weight: LayerNorm weight [channels]
        ln_bias: LayerNorm bias [channels]
        eps: LayerNorm epsilon
        
    Returns:
        (transposed_out, out, transposed_out) - matching the model return pattern
    """
    batch_size, channels, height, width = input_tensor.shape
    out_channels = channels
    seq_len = height * width
    kernel_size = weight.shape[2]  # 3
    
    # Create output tensors
    # Output shape after flatten and transpose: [batch, seq_len, channels]
    # For model return, we need: tmp_7 (transpose), tmp_10 (no transpose), tmp_9 (transpose)
    # tmp_7 = tmp_6.transpose(1, 2) -> [1, 256, 768] format (seq, batch, channels)
    # tmp_9 = tmp_8.transpose(0, 1) -> [256, 1, 768]
    # tmp_10 = tmp_8.transpose(0, 1) -> same as tmp_9
    
    # Create output tensor in [batch, seq_len, channels] format
    out = torch.empty(batch_size, seq_len, out_channels, 
                      dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For the transposed outputs, we need [seq_len, batch, channels] format
    # But we can compute this by storing differently
    out_transposed = torch.empty(seq_len, batch_size, out_channels,
                                  dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get strides
    input_stride = input_tensor.stride()
    weight_stride = weight.stride()
    residual_stride = residual.stride()
    out_stride = out.stride()
    
    # Grid configuration
    # We parallelize over batch * height * width positions
    grid = (batch_size, height, width)
    
    # Choose BLOCK_SIZE based on number of channels for optimal performance
    # For smaller channels (768), use smaller block
    # For larger channels (1024), use larger block for better efficiency
    block_size = 2048 if out_channels >= 1024 else 1024
    
    # Launch kernel
    fused_conv_add_layernorm_kernel[grid](
        input_tensor, weight, bias, residual,
        ln_weight, ln_bias,
        out_transposed, out,
        input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        residual_stride[0], residual_stride[1], residual_stride[2], residual_stride[3],
        out_stride[0], out_stride[1], out_stride[2],
        batch_size, channels, out_channels, height, width,
        kernel_size, 1, 1,  # padding, stride (from model: (1, 1), (1, 1), (1, 1))
        seq_len, eps,
        BLOCK_SIZE=block_size,
    )
    
    return out_transposed, out, out_transposed


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the computation pattern:
    1. Conv2D with bias
    2. Add residual
    3. Flatten spatial dims
    4. Transpose
    5. LayerNorm
    6. Two transpose(0, 1)
    
    Note: groups is determined by in_3.shape[0] (weight tensor's first dimension).
    For depthwise conv, groups = out_channels = in_channels = in_3.shape[0].
    """
    groups = in_3.shape[0]
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), groups)
    tmp_5 = conv2d + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (groups,), in_1, in_0, 1e-05)
    tmp_9 = tmp_8.transpose(0, 1)
    tmp_10 = tmp_8.transpose(0, 1)
    return tmp_7, tmp_10, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments needed for the replacement function.
    LayerNorm params are in_0 (bias) and in_1 (weight), conv params are in_2 (bias), in_3 (weight), in_4 (input).
    """
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    """
    Return the replacement function that implements the fused kernel.
    """
    return fused_conv_add_layernorm_wrapper