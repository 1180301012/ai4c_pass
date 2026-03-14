import torch
import triton
import triton.language as tl

# Simple pattern to test batch norm matching
def pattern(in_4, tmp_0, tmp_1, tmp_3, tmp_2):
    # Just match the batch norm operation exactly
    tmp_4 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_4

# Argument extraction
def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

# Optimized batch norm using Triton with autotuning
@triton.jit
def simple_bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    num_features,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with float32 precision explicitly
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Calculate feature index for BN parameters
    feature_idx = offsets % num_features
    
    # Load BN parameters with more precise masking
    mean = tl.load(mean_ptr + feature_idx, mask=feature_idx < num_features, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + feature_idx, mask=feature_idx < num_features, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + feature_idx, mask=feature_idx < num_features, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + feature_idx, mask=feature_idx < num_features, other=0.0).to(tl.float32)
    
    # Compute standard deviation safely
    std_inv = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transformation
    x_norm = (x - mean) * std_inv
    x_out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, x_out, mask=mask)

@torch.fx.wrap
def simple_bn_optimized(in_4, running_mean, running_var, weight, bias):
    # Ensure inputs are on the same device
    if running_mean.device != in_4.device:
        running_mean = running_mean.to(in_4.device)
    if running_var.device != in_4.device:
        running_var = running_var.to(in_4.device)
    if weight.device != in_4.device:
        weight = weight.to(in_4.device)
    if bias.device != in_4.device:
        bias = bias.to(in_4.device)
    
    batch_size, num_features, height, width = in_4.shape
    spatial_size = height * width
    n_elements = batch_size * num_features * spatial_size
    
    # Dynamic block size selection for better performance
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 10000:
        BLOCK_SIZE = 512  
    elif n_elements < 100000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_4, dtype=torch.float32)
    
    simple_bn_kernel[(num_programs,)](
        x_ptr=in_4,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        num_features=num_features,
        spatial_size=spatial_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_bn_optimized