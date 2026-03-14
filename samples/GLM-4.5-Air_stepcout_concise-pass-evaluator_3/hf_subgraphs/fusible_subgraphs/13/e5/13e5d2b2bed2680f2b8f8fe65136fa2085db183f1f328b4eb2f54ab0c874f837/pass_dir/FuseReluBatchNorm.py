import torch
import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    num_features,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and apply ReLU
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_relu = tl.maximum(x, 0.0)
    
    # Calculate indices for feature-wise params
    feature_idx = offsets % num_features
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + feature_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
    
    # BatchNorm computation: (x - mean) / sqrt(var + eps) * weight + bias
    var = running_var
    sqrt_var = tl.sqrt(var + eps)
    inv_std = 1.0 / sqrt_var
    
    x_normalized = (x_relu - running_mean) * inv_std
    out = x_normalized * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(x, running_mean, running_var, weight, bias):
    n_elements = x.numel()
    num_features = x.shape[-1]
    
    # Use block size that divides evenly into the problem size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        num_features=num_features,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_4, in_0, in_1, in_3, in_2):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

def replacement_func():
    return fused_relu_batch_norm