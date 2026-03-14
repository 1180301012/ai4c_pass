import torch
import triton
import triton.language as tl


@triton.jit
def fused_dwconv_add_bn_kernel(
    # Input and output pointers
    input_ptr, residual_ptr, weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    output_ptr,
    # Shape information
    batch_size, channels, height, width,
    # BN epsilon
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. Depthwise conv2d
    2. Add residual
    3. Batch normalization
    4. Adaptive avg pool to 1x1
    5. Flatten
    
    All in one kernel to minimize memory traffic and kernel launch overhead.
    """
    # Each program processes one channel
    channel_idx = tl.program_id(0)
    
    # Calculate offsets
    channel_offset = channel_idx * height * width
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight_bn = tl.load(weight_bn_ptr + channel_idx)
    bias_bn = tl.load(bias_bn_ptr + channel_idx) if bias_bn_ptr else 0.0
    
    # Compute batch norm scale and bias
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = weight_bn * inv_std
    bn_bias = bias_bn - weight_bn * running_mean * inv_std
    
    # Load conv weight (depthwise, shape: [C, 1, 7, 7])
    # For depthwise: output[c,h,w] = sum(weight[c,0,kh,kw] * input[c,h+kh-3,w+kw-3]) + bias[c]
    conv_bias = tl.load(bias_ptr + channel_idx) if bias_ptr else 0.0
    
    # Accumulator for adaptive avg pool (sum all values)
    sum_val = 0.0
    
    # Iterate over the spatial dimensions
    # depthwise conv with padding=3, stride=1, so input and output have same spatial size
    for h in range(height):
        for w in range(width):
            # Compute conv: 7x7 depthwise convolution
            conv_val = 0.0
            
            # Load the depthwise weight and compute convolution
            # Weight shape: [C, 1, 7, 7], iterate over kh, kw
            for kh in range(7):
                for kw in range(7):
                    # Input coordinates with padding (padding=3 means we pad the input)
                    # conv with padding=3, kernel=7, stride=1 gives same size output
                    # So input effective coordinates:
                    in_h = h + kh - 3
                    in_w = w + kw - 3
                    
                    # Check bounds (padding creates zeros outside)
                    if 0 <= in_h < height and 0 <= in_w < width:
                        input_offset = channel_idx * height * width + in_h * width + in_w
                        weight_offset = channel_idx * 49 + kh * 7 + kw  # 7*7=49
                        
                        inp_val = tl.load(input_ptr + input_offset)
                        wt_val = tl.load(weight_ptr + weight_offset)
                        conv_val += inp_val * wt_val
            
            conv_val += conv_bias
            
            # Load residual and add
            residual_offset = channel_idx * height * width + h * width + w
            residual_val = tl.load(residual_ptr + residual_offset)
            
            # Fused add + batch_norm
            # (conv_val + residual - mean) * scale + bn_bias
            bn_val = (conv_val + residual_val - running_mean) * scale + bn_bias
            
            # Accumulate for adaptive avg pool
            sum_val += bn_val
    
    # Compute average (adaptive avg pool to 1x1)
    num_elements = height * width
    avg_val = sum_val / num_elements
    
    # Store result (flattened: [batch_size, channels])
    # For batch=0 (or we process per-channel per-batch)
    # Actually we need to handle batch dimension properly
    # Let me fix this - we need to process all batches
    
    
@triton.jit
def fused_dwconv_add_bn_kernel_v2(
    # Input pointers
    input_ptr, residual_ptr, 
    # Conv weight and bias
    conv_weight_ptr, conv_bias_ptr,
    # BN parameters
    running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    # Output
    output_ptr,
    # Shape: (batch, channels, height, width)
    batch_stride_in, ch_stride_in, h_stride_in, w_stride_in,
    batch_stride_res, ch_stride_res, h_stride_res, w_stride_res,
    batch_stride_out, ch_stride_out,
    batch_size, channels, height, width,
    # BN epsilon
    eps: tl.constexpr,
    # Block config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused depthwise conv + add + batch_norm + adaptive_avg_pool + flatten.
    
    This kernel computes:
    out = adaptive_avg_pool2d(batch_norm(conv2d(input) + residual), 1).flatten(1)
    
    Strategy: Each program processes one output channel. We compute the full
    depthwise conv, add residual, apply batch norm, and accumulate for avg pool.
    """
    # Program 0 processes channel 0, program 1 processes channel 1, etc.
    pid = tl.program_id(0)
    channel_idx = pid
    
    if channel_idx >= channels:
        return
    
    # Load BN parameters for this channel
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight_bn = tl.load(weight_bn_ptr + channel_idx)
    bias_bn_val = tl.load(bias_bn_ptr + channel_idx) if bias_bn_ptr else 0.0
    
    # Compute BN fused parameters: y = (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = weight_bn * inv_std
    bias_fused = bias_bn_val - weight_bn * running_mean * inv_std
    
    # Conv bias
    conv_bias = tl.load(conv_bias_ptr + channel_idx) if conv_bias_ptr else 0.0
    
    # Process all batches for this channel
    for batch_idx in range(batch_size):
        # Accumulator for adaptive avg pool
        sum_val = 0.0
        
        # Depthwise conv: for each output position, compute 7x7 convolution
        # Input: [B, C, H, W], Weight: [C, 1, 7, 7], stride=1, padding=3
        for h in range(height):
            for w in range(width):
                # Compute depthwise conv at position (h, w)
                conv_val = 0.0
                
                # 7x7 kernel
                for kh in range(7):
                    for kw in range(7):
                        # Input position with padding
                        in_h = h + kh - 3
                        in_w = w + kw - 3
                        
                        # Boundary check (for padding)
                        if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                            # Input offset
                            inp_offset = batch_idx * batch_stride_in + \
                                        channel_idx * ch_stride_in + \
                                        in_h * h_stride_in + \
                                        in_w * w_stride_in
                            
                            # Weight offset for depthwise conv
                            wt_offset = channel_idx * 49 + kh * 7 + kw
                            
                            inp_val = tl.load(input_ptr + inp_offset)
                            wt_val = tl.load(conv_weight_ptr + wt_offset)
                            conv_val += inp_val * wt_val
                
                conv_val += conv_bias
                
                # Add residual
                res_offset = batch_idx * batch_stride_res + \
                            channel_idx * ch_stride_res + \
                            h * h_stride_res + \
                            w * w_stride_res
                res_val = tl.load(residual_ptr + res_offset)
                
                # Apply batch norm
                bn_val = conv_val * scale + (res_val - running_mean) * scale + bias_fused
                # Simplified: (conv + res - mean) * scale + bias_fused
                bn_val = (conv_val + res_val) * scale + bias_fused - running_mean * scale
                
                sum_val += bn_val
        
        # Adaptive avg pool: divide by total spatial elements
        num_spatial = height * width
        avg_val = sum_val / num_spatial
        
        # Store flattened output: [batch_size, channels]
        out_offset = batch_idx * batch_stride_out + channel_idx * ch_stride_out
        tl.store(output_ptr + out_offset, avg_val)


@torch.fx.wrap
def fused_dwconv_add_bn_adaptive_avg_pool_wrapper(
    input: torch.Tensor,
    residual: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight_bn: torch.Tensor,
    bias_bn: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Wrapper function that launches the fused Triton kernel.
    
    Computes: adaptive_avg_pool2d(batch_norm(conv2d(input) + residual, 1)).flatten(1)
    """
    batch_size, channels, height, width = input.shape
    
    # Output shape: [batch_size, channels] (flattened)
    output = torch.empty((batch_size, channels), dtype=input.dtype, device=input.device)
    
    # Configure kernel
    BLOCK_SIZE = 256
    
    # Launch kernel: one program per channel
    grid = (channels,)
    
    fused_dwconv_add_bn_kernel_v2[grid](
        input, residual,
        conv_weight, conv_bias,
        running_mean, running_var, weight_bn, bias_bn,
        output,
        # Strides
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
        output.stride(0), output.stride(1),
        batch_size, channels, height, width,
        eps,
        BLOCK_SIZE
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the pattern:
    tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (3, 3), (1, 1), 192)
    tmp_7 = in_7 + tmp_6
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    tmp_10 = tmp_9.flatten(1, -1)
    return tmp_10
    """
    # Conv2d - simplified without extra args
    # The model has stride=1, padding=3, dilation=1 but we skip these for matching
    tmp_6 = torch.nn.functional.conv2d(in_6, in_5, in_4)
    
    # Add residual
    tmp_7 = in_7 + tmp_6
    
    # Batch norm: batch_norm(input, running_mean, running_var, weight, bias, ...)
    # Parameters: batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    # in_0 = running_mean, in_1 = running_var, in_2 = bias, in_3 = weight
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Adaptive avg pool to 1x1
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    
    # Flatten
    tmp_10 = tmp_9.flatten(1, -1)
    
    return tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Extract arguments for the replacement function.
    
    in_0: running_mean
    in_1: running_var  
    in_2: bias (BN bias)
    in_3: weight (BN weight)
    in_4: conv_bias
    in_5: conv_weight
    in_6: input
    in_7: residual
    """
    return (in_6, in_7, in_5, in_4, in_0, in_1, in_3, in_2, 1e-05)


def replacement_func():
    return fused_dwconv_add_bn_adaptive_avg_pool_wrapper