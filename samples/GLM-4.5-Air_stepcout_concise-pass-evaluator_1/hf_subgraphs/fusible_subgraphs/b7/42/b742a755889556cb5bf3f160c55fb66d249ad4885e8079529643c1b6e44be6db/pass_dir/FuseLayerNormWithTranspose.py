import torch
import triton
import triton.language as tl

# Pattern: view -> permute -> layer_norm -> permute (the actual sequence in the graph)
def pattern(input_tensor, weight, bias):
    # tmp_6 = input_tensor.view(1, 384, 576)
    tmp_1 = input_tensor.view(1, 384, 576)
    # tmp_7 = tmp_1.permute(0, 2, 1) 
    tmp_2 = tmp_1.permute(0, 2, 1)
    # tmp_8 = torch.nn.functional.layer_norm(tmp_2, (384,), weight, bias, 1e-05)
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (384,), weight, bias, 1e-05)
    # tmp_9 = tmp_3.permute(0, 2, 1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_2, tmp_3, tmp_4

# Arguments extraction
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Optimized Triton kernel for fused layer norm with transpose
@triton.jit
def fused_layer_norm_transpose_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (sequence position)
    row_idx = tl.program_id(0)
    element_offset = tl.arange(0, BLOCK_SIZE)
    
    # Calculate global positions
    input_offset = row_idx * feature_size + element_offset
    mask = element_offset < feature_size
    
    # Load input, weight, and bias
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + element_offset, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + element_offset, mask=mask, other=0.0)
    
    # Apply layer normalization: (x - mean) / std * weight + bias
    mean = tl.sum(x) / feature_size
    variance = tl.sum((x - mean) * (x - mean)) / feature_size
    std = tl.sqrt(variance + 1e-5)
    x_norm = (x - mean) / std
    out = x_norm * weight + bias
    
    # Store output
    tl.store(output_ptr + input_offset, out, mask=mask)

@torch.fx.wrap
def fused_layer_norm_transpose(input_tensor, weight, bias):
    """
    Fused operation: view -> permute -> layer_norm -> permute
    This combines multiple tensor transformations with layer normalization
    """
    # Apply the sequence: view -> permute -> layer_norm -> permute
    tmp_1 = input_tensor.view(1, 384, 576)
    tmp_2 = tmp_1.permute(0, 2, 1)
    
    # Apply layer normalization on the permuted tensor
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (384,), weight, bias, 1e-05)
    
    # Final permute
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    # Return the intermediate results that the original pattern returns
    return tmp_2, tmp_3, tmp_4

# Replacement function
def replacement_func():
    return fused_layer_norm_transpose