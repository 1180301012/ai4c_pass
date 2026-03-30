import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, eps=1e-5):
    """Pattern: batch_norm followed by relu"""
    # Note: Using the exact parameters from the model - eps is different from model's 0.001
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=eps)
    relu_out = torch.nn.functional.relu(bn_out, inplace=False)
    return bn_out, relu_out

def replacement_args(x, running_mean, running_var, weight, bias, eps=0.001):
    return (x, running_mean, running_var, weight, bias, eps)

@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    batch_size,
    eps: tl.constexpr,
    BLOCK_SIZE_CHAN: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one spatial location across all channels
    batch_id = pid // (height * width)
    spatial_id = pid % (height * width)
    h = spatial_id // width
    w = spatial_id % width
    
    # Ensure we don't go out of bounds
    if batch_id >= batch_size or h >= height or w >= width:
        return
    
    # Load parameters for this batch channel combination
    chan_offset = batch_id * n_channels
    for c in range(0, n_channels, BLOCK_SIZE_CHAN):
        block_start = c
        block_end = min(c + BLOCK_SIZE_CHAN, n_channels)
        mask = c < n_channels
        
        # Load parameters
        weight = tl.load(weight_ptr + block_start, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + block_start, mask=mask, other=0.0)
        mean = tl.load(running_mean_ptr + block_start, mask=mask, other=0.0)
        var = tl.load(running_var_ptr + block_start, mask=mask, other=1.0)
        
        # Load input data
        x_offset = (batch_id * n_channels + block_start) * height * width + spatial_id
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Batch norm calculation: (x - mean) / sqrt(var + eps) * weight + bias
        var_eps = var + eps
        inv_std = tl.math.rsqrt(var_eps)
        normalized = (x_val - mean) * inv_std
        bn_val = normalized * weight + bias
        
        # ReLU activation
        relu_val = tl.math.maximum(bn_val, 0.0)
        
        # Store result
        out_offset = x_offset  # Same shape as input
        tl.store(out_ptr + out_offset, relu_val, mask=mask)

@torch.fx.wrap
def fused_bn_relu(x, running_mean, running_var, weight, bias, eps=0.001):
    """Fused batch normalization + ReLU operation"""
    batch_size, channels, height, width = x.shape
    n_elements = x.numel()
    
    # Choose block size based on channels for good occupancy
    if channels <= 64:
        BLOCK_SIZE_CHAN = 64
    elif channels <= 128:
        BLOCK_SIZE_CHAN = 32
    else:
        BLOCK_SIZE_CHAN = 16
    
    # Grid size: each program handles one spatial location across all channels
    grid_size = batch_size * height * width
    
    # Allocate output buffer
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_bn_relu_kernel[grid_size](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_channels=channels,
        height=height,
        width=width,
        batch_size=batch_size,
        eps=eps,
        BLOCK_SIZE_CHAN=BLOCK_SIZE_CHAN,
    )
    
    return out

def replacement_func():
    return fused_bn_relu