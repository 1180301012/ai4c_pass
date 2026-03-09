import torch
import triton
import triton.language as tl

# Pattern matching function for batch normalization
def pattern(input, running_mean, running_var, weight, bias):
    # The exact pattern that matches torch.nn.functional.batch_norm with specific parameters
    out = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=0.001)
    return out

# Argument extraction function
def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

# Optimized Triton kernel for batch normalization
@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    C,  # number of channels
    H,  # height
    W,  # width
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a single channel
    pid = tl.program_id(0)
    
    # Create coordinate for this channel
    base_offset = pid * (H * W)
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data (NHWC format)
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Load batch norm parameters (this is a per-channel operation)
    # Only load parameters if we have a valid channel index
    param_mask = pid < C
    if param_mask:
        running_mean_val = tl.load(running_mean_ptr + pid)
        running_var_val = tl.load(running_var_ptr + pid)
        weight_val = tl.load(weight_ptr + pid)
        bias_val = tl.load(bias_ptr + pid)
    else:
        # Fallback values for channels beyond the batch norm parameter size
        running_mean_val = 0.0
        running_var_val = 1.0
        weight_val = 1.0
        bias_val = 0.0
    
    # Compute batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    denom = tl.sqrt(running_var_val + eps)
    output_data = (input_data - running_mean_val) / denom * weight_val + bias_val
    
    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)

@torch.fx.wrap
def triton_batch_norm(input, running_mean, running_var, weight, bias):
    # Input tensor shape: [N, C, H, W]
    N, C, H, W = input.shape
    n_elements = N * C * H * W
    
    # Calculate block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (C * N,)  # One program per channel per batch
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch the kernel
    batch_norm_kernel[grid](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        C=C,
        H=H,
        W=W,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return triton_batch_norm