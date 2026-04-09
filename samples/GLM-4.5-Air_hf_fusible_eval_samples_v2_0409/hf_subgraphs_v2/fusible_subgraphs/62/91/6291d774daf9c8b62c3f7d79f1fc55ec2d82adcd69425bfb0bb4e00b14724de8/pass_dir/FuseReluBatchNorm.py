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
    FEATURE_BLOCK: tl.constexpr,
):
    """Fused ReLU + BatchNorm kernel for better GPU performance"""
    # Each program handles a portion of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input element
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute feature index for normalization parameters
    feat_idx = offsets % num_features
    
    # Load normalization parameters using feature-wise indexing
    # We load one feature at a time and broadcast within the warp
    running_mean = tl.load(running_mean_ptr + feat_idx, mask=feat_idx < num_features, other=0.0)
    running_var = tl.load(running_var_ptr + feat_idx, mask=feat_idx < num_features, other=1.0)
    weight = tl.load(weight_ptr + feat_idx, mask=feat_idx < num_features, other=1.0)
    bias = tl.load(bias_ptr + feat_idx, mask=feat_idx < num_features, other=0.0)
    
    # Apply batch normalization: (x - mean) * (weight / sqrt(var + eps)) + bias
    denom = tl.sqrt(running_var + eps)
    normalized = (x - running_mean) * (weight / denom) + bias
    
    # Apply ReLU activation
    result = tl.maximum(normalized, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(x, running_mean, running_var, weight, bias, eps=1e-05):
    """Fused ReLU + BatchNorm implementation using Triton"""
    n_elements = x.numel()
    num_features = running_mean.numel()
    
    # Optimal block size for GPU occupancy
    BLOCK_SIZE = 1024  # Standard block size for good GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure output has same dtype and device as input
    out = torch.empty_like(x)
    
    # Launch kernel
    relu_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        num_features=num_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        FEATURE_BLOCK=32,  # Number of features to process per warp
    )
    
    return out

def pattern(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    """Pattern to match ReLU followed by BatchNorm"""
    # ReLU operation
    relu_out = torch.nn.functional.relu(x, inplace=False)
    
    # BatchNorm operation
    batch_norm_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, weight, bias, training, momentum, eps)
    
    return batch_norm_out

def replacement_args(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    """Extract arguments for the replacement - focus on the ones we need for fusion"""
    return (x, running_mean, running_var, weight, bias, eps)

def replacement_func():
    """Return the fused ReLU + BatchNorm function"""
    return fused_relu_batch_norm