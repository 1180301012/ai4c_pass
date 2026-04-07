import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, eps=0.1, momentum=0.001):
    # Batch norm operation
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=eps, momentum=momentum)
    # ReLU operation
    result = torch.nn.functional.relu(out, inplace=False)
    return result

def replacement_args(x, running_mean, running_var, weight, bias, eps=0.1, momentum=0.001):
    return (x, running_mean, running_var, weight, bias, eps, momentum)

@triton.jit
def batchnorm_relu_kernel(
    x_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    out_ptr,
    n_channels,       # Feature dimension (channels)
    height,           # Spatial height  
    width,            # Spatial width
    batch_size,       # Batch size
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total elements and offsets
    total_elements = n_channels * height * width
    batch_offset = pid * total_elements
    channel_size = height * width
    
    # Load batch normalization parameters
    mean = tl.load(running_mean_ptr + pid).to(tl.float32)
    var = tl.load(running_var_ptr + pid).to(tl.float32)
    weight_val = tl.load(weight_ptr + pid).to(tl.float32)
    bias_val = tl.load(bias_ptr + pid).to(tl.float32)
    
    # Precompute normalization factors
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = weight_val * inv_std
    bias_scaled = bias_val - mean * scale
    
    # Load input and compute for each position
    for i in range(channel_size):
        offset = batch_offset + i
        if offset < total_elements:
            x_val = tl.load(x_ptr + offset).to(tl.float32)
            
            # Batch normalization: y = (x - mean) * (weight / sqrt(var + eps)) + bias
            bn_val = (x_val - mean) * scale + bias_scaled
            
            # ReLU activation: max(0, bn_val)
            relu_val = tl.maximum(bn_val, 0.0)
            
            # Store result
            tl.store(out_ptr + offset, relu_val.to(x_val.dtype()))

@torch.fx.wrap
def batchnorm_relu(x, running_mean, running_var, weight, bias, eps=0.1, momentum=0.001):
    # Get tensor shapes
    batch_size, n_channels, height, width = x.shape
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32) if x.dtype == torch.bfloat16 else torch.empty_like(x)
    
    # For different data types, we need to handle appropriately
    if x.dtype == torch.bfloat16:
        # Work with float32 for computation, then convert back
        x_float32 = x.to(torch.float32)
    else:
        x_float32 = x
    
    # Launch kernel
    grid = (batch_size,)
    batchnorm_relu_kernel[grid](
        x_ptr=x_float32,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        batch_size=batch_size,
        eps=eps,
    )
    
    # Convert back to original dtype if needed
    if x.dtype == torch.bfloat16:
        return out.to(torch.bfloat16)
    return out

def replacement_func():
    return batchnorm_relu