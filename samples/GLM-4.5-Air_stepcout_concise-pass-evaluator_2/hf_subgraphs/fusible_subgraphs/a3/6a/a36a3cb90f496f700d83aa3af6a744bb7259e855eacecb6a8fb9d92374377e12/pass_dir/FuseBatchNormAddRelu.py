import torch
import triton
import triton.language as tl

# Pattern matching function - focus on the core computation sequence
def pattern(in_4, dummy_0, dummy_1, dummy_3, dummy_2, in_5):
    # The batch_norm operation with the exact parameters from model.py
    tmp_4 = torch.nn.functional.batch_norm(in_4, dummy_0, dummy_1, dummy_3, dummy_2, False, 0.1, 1e-05)
    
    # The addition operation
    tmp_5 = in_5 + tmp_4
    
    # The ReLU operation
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    
    # Return tmp_6 for both subsequent mean operation and output
    return tmp_6, tmp_6

# Argument extraction function
def replacement_args(in_4, running_mean, running_var, weight, bias, in_5):
    return (in_4, running_mean, running_var, weight, bias, in_5)

# Triton kernel for fused BatchNorm + Add + ReLU
@triton.jit
def fused_bn_add_relu_kernel(
    x_ptr,           # Input tensor (in_4)
    y_ptr,           # Addend tensor (in_5) 
    mean_ptr,        # Running mean (BN parameter)
    var_ptr,         # Running variance (BN parameter)
    weight_ptr,      # Weight (BN parameter)
    bias_ptr,        # Bias (BN parameter)
    out_ptr,         # Output (tmp_6)
    n_elements,      # Total number of elements
    num_features,    # Number of features (C)
    spatial_size,    # Spatial size (H * W)
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with padding for out-of-bounds elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate spatial and feature indices for BN parameters
    spatial_idx = offsets // num_features  # Index in spatial dimension
    feature_idx = offsets % num_features   # Index in feature dimension
    
    # Load BN parameters (these are 1D vectors of size num_features)
    mean = tl.load(mean_ptr + feature_idx, mask=feature_idx < num_features, other=0.0)
    var = tl.load(var_ptr + feature_idx, mask=feature_idx < num_features, other=1.0)
    weight = tl.load(weight_ptr + feature_idx, mask=feature_idx < num_features, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, mask=feature_idx < num_features, other=0.0)
    
    # Apply BatchNorm normalization
    var_inv = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * var_inv
    
    # Scale and shift
    x_scaled = x_norm * weight + bias
    
    # Add the second input tensor
    x_added = x_scaled + y
    
    # Apply ReLU
    x_out = tl.where(x_added > 0, x_added, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, x_out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_bn_add_relu(in_4, running_mean, running_var, weight, bias, in_5):
    # Get tensor dimensions
    batch_size, num_features, height, width = in_4.shape
    spatial_size = height * width
    n_elements = batch_size * num_features * spatial_size
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_4)
    
    # Launch the kernel
    fused_bn_add_relu_kernel[(num_programs,)](
        x_ptr=in_4,
        y_ptr=in_5,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        num_features=num_features,
        spatial_size=spatial_size,
        momentum=0.1,      # From model.py
        eps=1e-05,        # From model.py
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out, out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_bn_add_relu