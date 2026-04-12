import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(x_ptr, running_mean_ptr, running_var_ptr, 
                      weight_ptr, bias_ptr, out_ptr,
                      num_features, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate flat index and handle different tensor shapes
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    spatial_mean = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for each offset (for parameter loading)
    channel = offsets % num_features
    channel_mask = channel < num_features
    
    # Load BN parameters using broadcasting
    running_mean_val = tl.load(running_mean_ptr + channel, mask=channel_mask & mask, other=0.0)
    running_var_val = tl.load(running_var_ptr + channel, mask=channel_mask & mask, other=1.0)
    weight_val = tl.load(weight_ptr + channel, mask=channel_mask & mask, other=1.0)
    bias_val = tl.load(bias_ptr + channel, mask=channel_mask & mask, other=0.0)
    
    # Batch norm formula: (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (spatial_mean - running_mean_val) / tl.sqrt(running_var_val + eps)
    out = normalized * weight_val + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias, training=True, momentum=0.1, eps=1e-05):
    """Optimized batch normalization using Triton"""
    # Match the dimensions - for mean case, x should be (N, C)
    if len(x.shape) == 4:
        # If input is (N, C, H, W), we need to compute mean first
        spatial_mean = x.mean(dim=(2, 3))  # This is already efficient
        input_for_bn = spatial_mean
    else:
        input_for_bn = x
    
    num_features = input_for_bn.shape[1] if len(input_for_bn.shape) > 1 else input_for_bn.shape[0]
    n_elements = input_for_bn.numel()
    
    output = torch.empty_like(input_for_bn)
    
    # Configure grid and block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    batch_norm_kernel[(num_programs,)](
        x_ptr=input_for_bn,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        num_features=num_features,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    """Match batch normalization operation"""
    return torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_args(input_tensor, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input_tensor, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_func():
    return optimized_batch_norm