import torch
import triton
import triton.language as tl

@triton.jit
def simple_bn_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    num_features,
    eps: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters
    running_mean = tl.load(running_mean_ptr + (offsets % num_features), mask=offsets % num_features < num_features, other=0.0)
    running_var = tl.load(running_var_ptr + (offsets % num_features), mask=offsets % num_features < num_features, other=1.0)
    weight = tl.load(weight_ptr + (offsets % num_features), mask=offsets % num_features < num_features, other=1.0)
    bias = tl.load(bias_ptr + (offsets % num_features), mask=offsets % num_features < num_features, other=0.0)
    
    # BatchNorm computation
    # Normalize: (x - running_mean) / sqrt(running_var + eps)
    variance = running_var + eps
    inv_std = tl.sqrt(tl.maximum(variance, 1e-5))
    normalized = (x - running_mean) * inv_std
    # Scale and shift: weight * normalized + bias
    out = weight * normalized + bias
    
    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_batch_norm(x, running_mean, running_var, weight, bias, training=True, momentum=0.1, eps=1e-05):
    # Determine tensor shapes
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Choose block size based on tensor size
    BLOCK_SIZE = 1024
    if n_elements < 1024:
        BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    simple_bn_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        y_ptr=y,
        n_elements=n_elements,
        num_features=C,
        eps=eps,
        training=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

def pattern(input, running_mean, running_var, weight, bias, training, momentum, eps):
    # Match the exact batch_norm call from the model
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_func():
    return simple_batch_norm