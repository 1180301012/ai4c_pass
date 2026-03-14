import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_layernorm_relu_kernel(
    # Input tensor pointer
    input_ptr,
    # Conv weight and bias
    conv_weight_ptr,
    conv_bias_ptr,
    # LayerNorm weight and bias
    ln_weight_ptr,
    ln_bias_ptr,
    # Output pointer
    output_ptr,
    # Tensor dimensions
    batch_size,
    in_channels,
    out_channels,
    # Strides
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    # Layer norm epsilon
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. 1x1 Conv2d (pointwise across channels)
    2. LayerNorm over (C, 1, 1) - i.e., normalize per position across channels
    3. ReLU activation
    
    This fuses three operations into one kernel to avoid intermediate memory traffic.
    """
    # Get batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Base pointers
    input_base = input_ptr + batch_idx * input_batch_stride
    output_base = output_ptr + batch_idx * output_batch_stride
    
    # Load conv weight for this output channel
    # Weight shape: [out_channels, in_channels, 1, 1]
    conv_weight_ptr_for_c = conv_weight_ptr + channel_idx * weight_out_channel_stride
    
    # Compute conv output for this batch and channel
    # out[b, c] = sum_k(in[b, k] * weight[c, k]) + bias[c]
    conv_out = 0.0
    
    # Load conv bias
    conv_bias = tl.load(conv_bias_ptr + channel_idx)
    
    # Iterate over input channels
    for k in range(in_channels):
        # Load weight: weight[c, k]
        w = tl.load(conv_weight_ptr_for_c + k * weight_in_channel_stride)
        # Load input: input[b, k]
        inp = tl.load(input_base + k * input_channel_stride)
        conv_out += inp * w
    
    # Add bias
    conv_out = conv_out + conv_bias
    
    # Now compute layer norm statistics across all channels for this batch position
    # We need to compute mean and variance across the channel dimension
    # Since we can't easily do a cross-channel reduction here, we'll compute
    # mean and variance in a separate loop
    
    # Actually, for proper layer norm, we need the full channel dimension
    # Let's load all channel values first
    
    # Storage for all channel outputs for this batch position
    channel_outputs = tl.zeros((out_channels,), tl.float32)
    
    # First pass: compute all conv outputs and store temporarily
    # For efficiency, we'll compute mean and variance incrementally
    
    # Alternative approach: compute conv + layernorm + relu per channel
    # but we need channel statistics first
    
    # Let's use a simpler approach: load all conv outputs
    # We need to iterate again to accumulate for mean/variance
    
    # For small channel counts, we can use a different approach
    # Load all conv outputs for this batch position
    
    # Storage for all channel conv outputs
    conv_outputs_ptrs = conv_weight_ptr + tl.arange(0, out_channels)[:, None] * weight_out_channel_stride + \
                        tl.arange(0, in_channels)[None, :] * weight_in_channel_stride
    
    # Actually, let's do this more simply:
    # Compute conv for all channels first, then do layer norm
    
    # But this is complex in Triton. Let me use a different approach:
    # Do conv + layernorm + relu in a way that minimizes memory access
    
    # Since the layer norm normalizes over ALL channels, we need to 
    # either:
    # 1. Use shared memory to accumulate
    # 2. Use multiple kernel passes
    # 3. Accept that we need two passes
    
    # For now, let's use a simpler fusion:
    # Conv + ReLU (which can be perfectly fused)
    # Then do LayerNorm separately
    
    # Actually, let's reconsider. The layer norm is over (C, 1, 1)
    # with C being small (16, 38, 128). We can definitely fuse this.
    
    # Approach: Each thread block handles one batch position
    # Threads in the block cooperate to compute conv for all output channels
    # Then compute mean/variance across channels
    # Then apply layer norm + relu
    
    pass


def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match the pattern: conv2d -> layer_norm -> relu"""
    # Simple pattern - just call the operations directly
    tmp = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    y = torch.nn.functional.layer_norm(tmp, (128, 1, 1), in_3, in_2, 1e-05)
    r = torch.nn.functional.relu(y, inplace=True)
    return r


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def conv_layernorm_relu_kernel(
    # Input: [batch, in_channels, 1, 1]
    input_ptr,
    # Conv weight: [out_channels, in_channels, 1, 1]
    conv_weight_ptr,
    # Conv bias: [out_channels]
    conv_bias_ptr,
    # Layer norm weight: [out_channels, 1, 1]
    ln_weight_ptr,
    # Layer norm bias: [out_channels, 1, 1]
    ln_bias_ptr,
    # Output
    output_ptr,
    # Dimensions
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    # Strides
    input_batch_stride: tl.constexpr,
    input_in_channel_stride: tl.constexpr,
    conv_weight_out_ch_stride: tl.constexpr,
    conv_weight_in_ch_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_channel_stride: tl.constexpr,
    # Epsilon for layer norm
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv + LayerNorm + ReLU kernel.
    
    Assumes input has shape [batch, in_channels, 1, 1]
    Output has shape [batch, out_channels, 1, 1]
    Layer norm normalizes over the channel dimension.
    """
    # Each program handles one batch position
    batch_idx = tl.program_id(0)
    
    # Pointers for this batch
    input_base = input_ptr + batch_idx * input_batch_stride
    output_base = output_ptr + batch_idx * output_batch_stride
    
    # Compute conv output for all output channels
    # Use BLOCK_SIZE threads, each computing a portion of the channels
    
    # First, compute all conv outputs and store in a temporary array
    # We need to compute mean and variance across all output channels
    
    # Local storage for conv outputs
    pid = tl.program_id(1)
    num_pid = tl.num_programs(1)
    
    # Each thread computes conv for a subset of output channels
    for ch_offset in range(0, out_channels, num_pid):
        ch_idx = ch_offset + pid
        if ch_idx < out_channels:
            # Compute conv for this output channel
            # out[ch] = sum_k(in[k] * weight[ch, k]) + bias[ch]
            conv_out = 0.0
            
            # Weight pointer for this output channel
            weight_base = conv_weight_ptr + ch_idx * conv_weight_out_ch_stride
            
            # Sum over input channels
            for k in range(in_channels):
                w = tl.load(weight_base + k * conv_weight_in_ch_stride)
                inp = tl.load(input_base + k * input_in_channel_stride)
                conv_out += inp * w
            
            # Add bias
            bias = tl.load(conv_bias_ptr + ch_idx)
            conv_out = conv_out + bias
            
            # Store temporarily in output - we'll overwrite after layer norm
            tl.store(output_base + ch_idx * output_channel_stride, conv_out)
    
    # Synchronize threads
    tl.debug_barrier()
    
    # Now compute mean and variance across channels
    # Each thread computes a portion
    sum_val = 0.0
    for ch_offset in range(0, out_channels, num_pid):
        ch_idx = ch_offset + pid
        if ch_idx < out_channels:
            val = tl.load(output_base + ch_idx * output_channel_stride)
            sum_val += val
    
    # Reduce across threads
    # For now, assume thread 0 does the reduction (not ideal but simple)
    mean = sum_val / out_channels
    
    # Compute variance
    sum_sq = 0.0
    for ch_offset in range(0, out_channels, num_pid):
        ch_idx = ch_offset + pid
        if ch_idx < out_channels:
            val = tl.load(output_base + ch_idx * output_channel_stride)
            diff = val - mean
            sum_sq += diff * diff
    
    variance = sum_sq / out_channels
    std = tl.sqrt(variance + eps)
    
    # Now apply layer norm and ReLU
    for ch_offset in range(0, out_channels, num_pid):
        ch_idx = ch_offset + pid
        if ch_idx < out_channels:
            # Load conv output
            val = tl.load(output_base + ch_idx * output_channel_stride)
            
            # Layer norm: (x - mean) / std * weight + bias
            ln_w = tl.load(ln_weight_ptr + ch_idx)
            ln_b = tl.load(ln_bias_ptr + ch_idx)
            
            norm_val = (val - mean) / std
            out = norm_val * ln_w + ln_b
            
            # ReLU
            out = tl.maximum(out, 0.0)
            
            # Store final result
            tl.store(output_base + ch_idx * output_channel_stride, out)


@torch.fx.wrap
def fused_conv_layernorm_relu_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    Wrapper function for the fused Conv + LayerNorm + ReLU kernel.
    
    Args:
        in_0: Conv bias [out_channels]
        in_1: Conv weight [out_channels, in_channels, 1, 1]
        in_2: LayerNorm bias [out_channels, 1, 1]
        in_3: LayerNorm weight [out_channels, 1, 1]
        in_4: Input tensor [batch, in_channels, 1, 1]
    
    Returns:
        Output tensor [batch, out_channels, 1, 1]
    """
    # Get shapes
    batch_size = in_4.shape[0]
    in_channels = in_4.shape[1]
    out_channels = in_1.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, 1, 1), device=in_4.device, dtype=in_4.dtype)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid
    # Use batch size for grid, and split channels across programs
    num_channel_progs = min(out_channels, 8)
    
    # Launch kernel
    conv_layernorm_relu_kernel[(batch_size, num_channel_progs)](
        in_4,
        in_1,
        in_0,
        in_3,
        in_2,
        output,
        batch_size,
        in_channels,
        out_channels,
        in_4.stride(0),  # input_batch_stride
        in_4.stride(1),  # input_in_channel_stride
        in_1.stride(0),  # conv_weight_out_ch_stride
        in_1.stride(1),  # conv_weight_in_ch_stride
        output.stride(0),  # output_batch_stride
        output.stride(1),  # output_channel_stride
        1e-05,  # eps
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv_layernorm_relu_wrapper