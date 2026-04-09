import torch
import triton
import triton.language as tl
import math

@triton.jit
def linear_sigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_features, out_features
):
    # Each program handles one output element
    batch_idx = tl.program_id(0) // out_features
    feature_idx = tl.program_id(0) % out_features
    
    # Early exit if out of bounds
    if batch_idx >= batch_size:
        return
    
    # Compute dot product for this batch and feature
    sum_val = 0.0
    for k in range(in_features):
        # Load input and weight
        offset_in = batch_idx * in_features + k
        offset_weight = feature_idx * in_features + k
        
        x_val = tl.load(x_ptr + offset_in)
        w_val = tl.load(weight_ptr + offset_weight)
        
        sum_val = sum_val + x_val * w_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + feature_idx)
    linear = sum_val + bias_val
    
    # Sigmoid activation
    sigmoid_result = 1.0 / (1.0 + tl.exp(-linear))
    
    # Store result
    output_offset = batch_idx * out_features + feature_idx
    tl.store(out_ptr + output_offset, sigmoid_result)

@torch.fx.wrap
def fused_linear_sigmoid(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    
    # Allocate output
    out = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
    
    # Create 1D grid: one program per output element
    grid_size = batch_size * out_features
    
    # Launch kernel with simple 1D grid (wrap in tuple)
    linear_sigmoid_kernel[(grid_size,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features
    )
    
    return out

def pattern(x, weight, bias):
    linear = torch.nn.functional.linear(x, weight, bias)
    tmp_3 = torch.sigmoid(linear)
    return tmp_3

def replacement_args(x, weight, bias):
    return (x, weight, bias)

def replacement_func():
    return fused_linear_sigmoid