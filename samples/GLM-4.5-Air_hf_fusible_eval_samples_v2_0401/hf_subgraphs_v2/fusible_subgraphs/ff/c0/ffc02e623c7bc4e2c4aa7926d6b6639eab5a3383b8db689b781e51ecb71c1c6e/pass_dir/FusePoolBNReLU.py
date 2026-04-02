import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    # Match the adaptive_avg_pool2d + batch_norm + relu pattern
    # Note: These mirror the operations in model.py exactly
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True)
    return tmp_6, tmp_8

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_pool_bn_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_batch, n_channels,
    height, width,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= n_batch * n_channels:
        return
    
    # Extract batch and channel indices
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Compute spatial mean for this batch and channel
    spatial_sum = 0.0
    spatial_count = 0
    
    for h in range(height):
        for w in range(width):
            x_offset = (batch_idx * n_channels * height * width + 
                       channel_idx * height * width + h * width + w)
            x_val = tl.load(x_ptr + x_offset)
            spatial_sum += float(x_val)
            spatial_count += 1
    
    # Compute mean
    spatial_mean = spatial_sum / spatial_count
    
    # Load BN parameters
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    denom = tl.sqrt(var + EPS)
    normalized = (spatial_mean - mean) / denom
    bn_output = normalized * weight_val + bias_val
    
    # Apply ReLU
    relu_output = max(0.0, float(bn_output))
    
    # Store results - mean is stored at [batch_idx, channel_idx, 0, 0]
    mean_offset = (batch_idx * n_channels + channel_idx)
    relu_offset = mean_offset
    
    tl.store(out_ptr + mean_offset, spatial_mean)
    tl.store(out_ptr + relu_offset + (n_batch * n_channels), relu_output)

@torch.fx.wrap
def fused_pool_bn_relu(x, running_mean, running_var, weight, bias):
    batch, channels, height, width = x.shape
    
    # Create output tensors
    means = torch.empty(batch, channels, device=x.device, dtype=x.dtype)
    relu_out = torch.empty(batch, channels, device=x.device, dtype=x.dtype)
    
    # Create combined output buffer for kernel writing
    # First half: means, second half: relu_out
    buffer_size = batch * channels * 2
    buffer = torch.empty(buffer_size, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    EPS = 1e-05  # Matches the training eps in the original batch_norm
    
    # Calculate grid size
    grid_size = batch * channels
    
    fused_pool_bn_relu_kernel[grid_size](
        x_ptr=x.reshape(-1),
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=buffer,
        n_batch=batch,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=EPS
    )
    
    # Copy results from buffer to separate tensors
    means_copy = buffer[:batch * channels].copy()
    relu_copy = buffer[batch * channels:].copy()
    
    # Reshape to proper dimensions
    means_final = means_copy.reshape(batch, channels)
    relu_final = relu_copy.reshape(batch, channels)
    
    return means_final.reshape(batch, channels, 1, 1), relu_final.reshape(batch, channels, 1, 1)

def replacement_func():
    return fused_pool_bn_relu