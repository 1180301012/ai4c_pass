import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches: linear transformation followed by sigmoid activation
    This pattern appears in all the target graphs as:
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused linear-sigmoid kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_sigmoid_kernel(
    bias_ptr,           # Pointer to bias tensor [out_features]
    weight_ptr,         # Pointer to weight tensor [out_features, in_features] 
    input_ptr,          # Pointer to input tensor [batch_size, in_features]
    output_ptr,         # Pointer to output tensor [batch_size, out_features]
    batch_size,         # Batch size
    in_features,        # Input features dimension
    out_features,       # Output features dimension
):
    # Each kernel computes one output element
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Check if this kernel should be active
    if batch_idx >= batch_size or feature_idx >= out_features:
        return
    
    # Compute dot product for this (batch, feature) pair
    acc = 0.0
    for k in range(in_features):
        # Load input, weight, and bias for current features
        input_val = tl.load(input_ptr + batch_idx * in_features + k)
        weight_val = tl.load(weight_ptr + feature_idx * in_features + k)
        acc += input_val.to(dtype=tl.float32) * weight_val.to(dtype=tl.float32)
    
    # Load bias and add to result
    bias_val = tl.load(bias_ptr + feature_idx)
    result = acc + bias_val.to(dtype=tl.float32)
    
    # Apply sigmoid and store
    output_sigmoid = 1.0 / (1.0 + tl.exp(-result))
    tl.store(output_ptr + batch_idx * out_features + feature_idx, output_sigmoid)

@torch.fx.wrap
def fused_linear_sigmoid_torch(in_0, in_1, in_2):
    """Torch wrapper for fused linear-sigmoid kernel"""
    # Handle different input shapes
    if in_2.dim() == 2:
        batch_size, in_features = in_2.shape
    else:
        # Handle case where batch_size is 1 (shape [1, features])
        batch_size = 1
        in_features = in_2.shape[-1]
    
    out_features = in_0.shape[0]
    
    # Create output tensor with same dtype as input
    output = torch.empty(batch_size, out_features, dtype=in_2.dtype, device=in_2.device)
    
    # Calculate 2D grid dimensions (one kernel per output element)
    grid = (batch_size, out_features)
    
    # Launch kernel with 2D grid
    fused_linear_sigmoid_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
    )
    
    return output

def replacement_func():
    """Return the fused linear-sigmoid function"""
    return fused_linear_sigmoid_torch