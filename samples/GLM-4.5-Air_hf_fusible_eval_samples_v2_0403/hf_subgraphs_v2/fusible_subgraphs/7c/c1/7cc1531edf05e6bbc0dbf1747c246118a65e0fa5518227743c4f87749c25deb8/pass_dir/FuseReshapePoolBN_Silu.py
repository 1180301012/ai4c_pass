import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    h_in,
    w_in,
    eps: tl.constexpr,
):
    """
    Fused kernel for: reshape + avg_pool2d + batch_norm + silu
    
    - Input: (batch_size, channels, h_in*2, w_in*2) = (4, 512, 16, 16)
    - Output: (batch_size, channels, h_in, w_in) = (4, 512, 8, 8)
    - Pools 2x2 regions in the last two dimensions
    """
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Each program handles h_in x w_in output positions
    # We use block indexing for spatial dimensions
    hw_in = h_in * w_in
    hw_in_block = tl.program_id(2)
    
    # Compute the output position
    out_h = hw_in_block // w_in
    out_w = hw_in_block % w_in
    
    # Compute output position in original tensor
    row_offset = batch_idx * channels * h_in * w_in + channel_idx * h_in * w_in + out_h * w_in + out_w
    row_offset_2x = batch_idx * channels * (h_in * 2) * (w_in * 2) + channel_idx * (h_in * 2) * (w_in * 2)
    
    # Load and sum the 2x2 region for avg pooling
    h0 = 2 * out_h
    w0 = 2 * out_w
    
    sum_val = 0.0
    for dh in tl.static_range(2):
        for dw in tl.static_range(2):
            h = h0 + dh
            w = w0 + dw
            flat_idx = row_offset_2x + h * (w_in * 2) + w
            x = tl.load(input_ptr + flat_idx)
            sum_val = sum_val + x
    
    # Average pooling
    pooled = sum_val / 4.0
    
    # Batch norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Batch normalization
    x = pooled - running_mean
    x = x / tl.sqrt(running_var + eps)
    x = x * weight + bias
    
    # SiLU activation: x / (1 + exp(-x))
    exp_neg_x = tl.exp(-x)
    out = x / (1.0 + exp_neg_x)
    
    # Store output
    tl.store(out_ptr + row_offset, out)


@torch.fx.wrap
def fused_kernel(
    input_tensor,
    running_mean,
    running_var,
    weight,
    bias,
    eps=1e-5,
):
    """
    Fused kernel wrapper that handles reshape + avg_pool2d + batch_norm + silu
    
    This is the main entry point for the optimized computation.
    """
    # Input shape: (4, 128, 256) from weight_meta
    # After reshape: (1, 512, 16, 16)
    # Note: batch size is 4 (from in_4.shape[0])
    
    batch_size = input_tensor.shape[0]
    channels = input_tensor.shape[1]
    h_in_2 = input_tensor.shape[2]  # This is 16
    w_in_2 = input_tensor.shape[3]  # This is 16
    h_in = h_in_2 // 2  # 8 after pooling
    w_in = w_in_2 // 2  # 8 after pooling
    
    # Output shape: (4, 512, 8, 8)
    out_shape = (batch_size, channels, h_in, w_in)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = (batch_size, channels, h_in * w_in)
    fused_kernel_kernel_autotuned[grid](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        batch_size,
        channels,
        h_in,
        w_in,
        eps,
    )
    
    return out


def pattern(input_tensor, running_mean, running_var, weight, bias):
    """
    Pattern that matches the computation:
    - input_tensor reshape(1, 512, 16, 16)
    - avg_pool2d with kernel=2, stride=2
    - batch_norm
    - silu
    """
    # Reshape
    reshaped = input_tensor.reshape(1, 512, 16, 16)
    
    # Avg pool 2d
    pooled = torch.nn.functional.avg_pool2d(reshaped, 2, 2, 0, False, True, None)
    
    # Batch norm (training=False, using running stats)
    normalized = torch.nn.functional.batch_norm(
        pooled, running_mean, running_var, weight, bias, False, 0.1, 1e-5
    )
    
    # SiLU activation
    activated = torch.nn.functional.silu(normalized, inplace=True)
    
    return activated


def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    """
    Extract arguments for the replacement kernel.
    """
    return (input_tensor, running_mean, running_var, weight, bias)


def replacement_func():
    """
    Return the optimized kernel function.
    """
    return fused_kernel