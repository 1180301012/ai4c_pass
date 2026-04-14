import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Thread offset
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load running statistics and parameters
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Batch normalization: y = (x - running_mean) / sqrt(running_var + epsilon) * weight + bias
    # Perform computation in higher precision for numerical stability
    normalized = (x.to(tl.float32) - running_mean.to(tl.float32)) / tl.sqrt(running_var.to(tl.float32) + epsilon)
    out = (normalized * weight.to(tl.float32) + bias.to(tl.float32)).to(x.dtype)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, epsilon=1e-05):
    # Since we're in inference mode (training=False), we use running statistics
    # which are already provided, so momentum is not used
    
    # Only apply optimization for 1D tensors (single feature vector)
    # For multi-dimensional tensors, return original computation
    if len(x.shape) > 1:
        # Fallback to original batch_norm for multi-dimensional inputs
        return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, epsilon)
    
    # Get number of elements (for 1D tensor)
    n_elements = x.shape[-1]
    
    # Choose optimal block size based on feature dimension
    if n_elements >= 512:
        BLOCK_SIZE = 1024
    elif n_elements >= 256:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output buffer
    out = torch.empty_like(x)
    
    # Apply batch norm kernel for 1D tensor
    batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        epsilon=epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_7, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

def replacement_func():
    return optimized_batch_norm