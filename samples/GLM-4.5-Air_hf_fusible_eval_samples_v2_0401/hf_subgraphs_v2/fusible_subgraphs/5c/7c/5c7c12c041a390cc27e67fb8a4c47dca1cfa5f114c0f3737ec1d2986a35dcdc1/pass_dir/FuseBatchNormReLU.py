import torch
import triton
import triton.language as tl

@triton.jit
def fused_batch_norm_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    batch_size,
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    pid = tl.program_id(0)
    
    # Bounds checking
    if pid >= n_channels:
        return
    
    # Load parameters for this channel
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    
    # Precompute normalization constants
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std
    scale = weight * inv_std
    bias_shifted = bias - mean * scale
    
    # Process all batches and spatial locations for this channel
    total_elements = batch_size * height * width
    
    for idx in range(0, total_elements, BLOCK_SIZE):
        # Compute offsets
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        # Load input data with broadcasting across batches
        input_data = tl.load(x_ptr + pid + offsets * n_channels, mask=mask, other=0.0)
        
        # Apply batch normalization
        normalized = input_data * scale + bias_shifted
        
        # Apply ReLU activation
        output_data = tl.maximum(normalized, 0.0)
        
        # Store output
        tl.store(out_ptr + pid + offsets * n_channels, output_data, mask=mask)

@torch.fx.wrap
def fused_batch_norm_relu(x, running_mean, running_var, weight, bias, momentum=0.1, eps=0.001):
    """
    Fused batch normalization with ReLU activation
    """
    # Get tensor shapes
    batch_size, n_channels, height, width = x.shape
    
    # Ensure tensors are on the same device and contiguous
    x = x.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Set up grid and launch kernel
    grid = (n_channels,)
    BLOCK_SIZE = 1024  # Optimal for GPU
    
    fused_batch_norm_relu_kernel[grid](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        batch_size=batch_size,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Pattern matching function
def pattern(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001):
    """
    Match batch_norm followed by relu pattern
    """
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    relu_out = torch.nn.functional.relu(bn_out, inplace=False)
    return relu_out

# Argument extraction function
def replacement_args(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001):
    return (x, running_mean, running_var, weight, bias, momentum, eps)

# Replacement function (returns function reference)
def replacement_func():
    return fused_batch_norm_relu