import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Matches the computation pattern: linear -> transpose -> element-wise multiplication
    This pattern appears in all the provided graph examples
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Basic Triton kernel that fuses linear, transpose, and multiplication
@triton.jit
def fused_linear_transpose_mul_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    mul_ptr,
    out_ptr,
    batch_size,
    input_dim,
    output_dim,
):
    # Program id: handles single element computation
    elem_idx = tl.program_id(0)
    
    # Calculate which batch and element this program handles
    batch_idx = elem_idx // (input_dim * output_dim)
    if batch_idx >= batch_size:
        return
    
    elem_idx_in_batch = elem_idx % (input_dim * output_dim)
    i = elem_idx_in_batch // output_dim  # input dimension (row in output)
    j = elem_idx_in_batch % output_dim   # output dimension (column in output)
    
    # Calculate offsets for this batch
    bias_start = batch_idx * output_dim
    input_start = batch_idx * input_dim
    mul_start = batch_idx * input_dim * output_dim
    out_start = batch_idx * input_dim * output_dim
    
    # Compute linear: output = bias + sum(input * weight.T)
    linear_result = 0.0
    
    # Load bias for this output feature (assuming valid access)
    bias_val = tl.load(bias_ptr + bias_start + j)
    linear_result += bias_val
    
    # Sum over input dimension: input @ weight.T
    for k in range(input_dim):
        # Load input value for this input feature
        input_val = tl.load(input_ptr + input_start + k)
        
        # Load weight value: original weight[j, k] becomes weight[k, j] in transpose
        weight_val = tl.load(weight_ptr + j * input_dim + k)
        
        linear_result += input_val * weight_val
    
    # Load multiplication coefficient for transposed position
    # Original mul has shape [batch, input_dim, output_dim]
    # For transpose, we access at [batch, i, j]
    mul_val = tl.load(mul_ptr + mul_start + i * output_dim + j)
    
    # Apply element-wise multiplication
    result = linear_result * mul_val
    
    # Store final result
    store_offset = out_start + i * output_dim + j
    tl.store(out_ptr + store_offset, result)

# Triton function to handle different batch sizes and tensor shapes
@torch.fx.wrap
def triton_fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    Fused Triton kernel combining linear transformation, transpose, and element-wise multiplication
    """
    batch_size, in_features, out_features = in_2.shape[0], in_2.shape[1], in_1.shape[1]
    
    # Output shape: (batch_size, in_features, out_features) for transposed result
    output_shape = (batch_size, in_features, out_features)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid size: one program per output element
    total_elements = batch_size * in_features * out_features
    grid = (total_elements,)
    
    # Launch simple kernel
    fused_linear_transpose_mul_kernel[grid](
        in_0,           # bias
        in_1,           # weight  
        in_2,           # input
        in_3,           # multiplication coefficients
        out,            # output
        batch_size,
        in_features,    # input_dim (features dimension, e.g., 196)
        out_features    # output_dim (features dimension, e.g., 196)
    )
    
    return out

# Replacement function (must return function reference)
def replacement_func():
    return triton_fused_linear_transpose_mul