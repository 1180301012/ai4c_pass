import torch
import triton
import triton.language as tl
import math

def pattern(x, running_mean, running_var, weight, bias):
    relu_out = torch.nn.functional.relu(x, inplace=False)
    bn_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return bn_out

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_relu_batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_features,
    n_elements,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature dimension
    feat_idx = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load parameters for this feature
    running_mean_val = tl.load(running_mean_ptr + feat_idx, mask=feat_idx < n_features)
    running_var_val = tl.load(running_var_ptr + feat_idx, mask=feat_idx < n_features)
    weight_val = tl.load(weight_ptr + feat_idx, mask=feat_idx < n_features)
    bias_val = tl.load(bias_ptr + feat_idx, mask=feat_idx < n_features)
    
    # Compute mean and variance correction
    denom = tl.sqrt(running_var_val + eps)
    inv_denom = 1.0 / denom
    normalized = (running_mean_val - running_mean_val) * inv_denom + bias_val  # mean=0 for training=False
    
    # Load input data and compute ReLU + BatchNorm
    x = tl.load(x_ptr + offsets + feat_idx * n_elements, mask=mask)
    relu_out = tl.maximum(x, 0.0)
    bn_out = relu_out * weight_val + normalized
    
    # Store result
    tl.store(out_ptr + offsets, bn_out, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(x, running_mean, running_var, weight, bias):
    n_features = running_mean.numel()
    n_elements = x.numel() // n_features
    
    out = torch.empty_like(x)
    
    # Optimize block size based on input size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (n_features, num_blocks,)
    
    fused_relu_batch_norm_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        n_features,
        n_elements,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_batch_norm