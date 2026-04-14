import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the sequence: ReLU → BatchNorm → Dropout
    """
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

# Triton kernel for fused ReLU + BatchNorm
@triton.jit
def fused_relu_batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    num_features,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters from CPU (they need to be on GPU)
    # We'll load the entire parameter vectors once per thread group
    mean_idx = tl.program_id(0) % (num_features // BLOCK_SIZE + 1)
    weight_idx = tl.program_id(0) % (num_features // BLOCK_SIZE + 1)
    bias_idx = tl.program_id(0) % (num_features // BLOCK_SIZE + 1)
    var_idx = tl.program_id(0) % (num_features // BLOCK_SIZE + 1)
    
    # Load batch norm parameters for the current feature dimension
    feature_offset = ((tl.program_id(0) * BLOCK_SIZE) // n_elements * num_features) % num_features
    running_mean = tl.load(running_mean_ptr + feature_offset, mask=feature_offset < num_features, other=0.0)
    running_var = tl.load(running_var_ptr + feature_offset, mask=feature_offset < num_features, other=1.0)
    weight = tl.load(weight_ptr + feature_offset, mask=feature_offset < num_features, other=1.0)
    bias = tl.load(bias_ptr + feature_offset, mask=feature_offset < num_features, other=0.0)
    
    # BatchNorm computation: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    x_normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    x_batch_norm = x_normalized * weight + bias
    
    # ReLU activation
    out = tl.maximum(x_batch_norm, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper for device transfers and kernel launch
@torch.fx.wrap
def fused_relu_batch_norm_gpu(in_0, in_1, in_2, in_3, in_4):
    # Move batch norm parameters to GPU if not already there
    device = in_4.device
    running_mean = in_0.to(device)
    running_var = in_1.to(device) 
    weight = in_3.to(device)
    bias = in_2.to(device)
    
    # Get tensor dimensions
    x = in_4
    n_elements = x.numel()
    num_features = in_0.shape[0]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Set up kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_relu_batch_norm_kernel[(num_programs,)](
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

# Alternative optimized CPU implementation when input is on CPU
@torch.fx.wrap
def fused_relu_batch_norm_cpu(in_0, in_1, in_2, in_3, in_4):
    # All tensors on CPU, use regular torch operations but optimized
    # Apply ReLU and BatchNorm in one fused operation
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    
    # Manual batch norm computation to avoid overhead
    running_mean = in_0
    running_var = in_1
    weight = in_3
    bias = in_2
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    x_normalized = (tmp_4 - running_mean) / torch.sqrt(running_var + 1e-05)
    x_batch_norm = x_normalized * weight + bias
    
    # ReLU
    out = torch.relu(x_batch_norm)
    
    return out

# Replacement function (returns the dispatch wrapper)
def replacement_func():
    return fused_relu_batch_norm_gpu