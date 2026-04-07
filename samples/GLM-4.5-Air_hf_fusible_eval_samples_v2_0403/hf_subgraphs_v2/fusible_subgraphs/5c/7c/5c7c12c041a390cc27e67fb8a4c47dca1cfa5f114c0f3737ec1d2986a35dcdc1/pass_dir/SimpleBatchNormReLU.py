import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    # Batch norm operation
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=0.1, momentum=0.001)
    # ReLU operation
    result = torch.nn.functional.relu(out, inplace=False)
    return result

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def simple_batchnorm_relu_kernel(
    x_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    out_ptr,
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Get per-thread offset
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load normalization parameters (for simplicity, use first element per thread)
    mean = tl.load(running_mean_ptr + (pid % running_mean_ptr.shape[0])).to(tl.float32)
    var = tl.load(running_var_ptr + (pid % running_var_ptr.shape[0])).to(tl.float32)
    weight_val = tl.load(weight_ptr + (pid % weight_ptr.shape[0])).to(tl.float32)
    bias_val = tl.load(bias_ptr + (pid % bias_ptr.shape[0])).to(tl.float32)
    
    # Batch normalization with ReLU
    eps = 0.1
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * weight_val + bias_val
    result = tl.maximum(y, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)

@torch.fx.wrap
def simple_batchnorm_relu(x, running_mean, running_var, weight, bias):
    # Flatten input for simplicity
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    
    # Create output
    out = torch.empty_like(x_flat)
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, 128),)
    simple_batchnorm_relu_kernel[grid](
        x_ptr=x_flat,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    # Reshape back to original
    return out.reshape(original_shape)

def replacement_func():
    return simple_batchnorm_relu