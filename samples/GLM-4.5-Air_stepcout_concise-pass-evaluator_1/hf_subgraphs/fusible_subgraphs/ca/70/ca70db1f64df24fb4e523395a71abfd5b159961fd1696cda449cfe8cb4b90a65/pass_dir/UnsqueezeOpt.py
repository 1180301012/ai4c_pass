import torch
import triton
import triton.language as tl

# Pattern matching function for unsqueeze operation
def pattern(input_tensor):
    """
    Pattern: input_tensor.unsqueeze(-2)
    This corresponds to: tmp_13 = tmp_6.unsqueeze(-2)
    - input_tensor: [300, 256] (tmp_6)
    - Output: [300, 1, 256] (tmp_13)
    """
    return input_tensor.unsqueeze(-2)

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for optimized unsqueeze operation
@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_features,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """
    Optimized unsqueeze operation to add a dimension at position -2
    [batch, features] -> [batch, 1, features]
    """
    # Program identifiers
    batch_pid = tl.program_id(0)
    feat_pid = tl.program_id(1)
    
    batch_idx = batch_pid
    feat_idx = feat_pid
    
    if batch_idx >= batch_size or feat_idx >= input_features:
        return
    
    # Load input element
    input_val = tl.load(input_ptr + batch_idx * input_features + feat_idx, other=0.0)
    
    # Store to output with inserted dimension
    output_base = output_ptr + batch_idx * (1 * input_features) + 1 * input_features
    tl.store(output_base + feat_idx, input_val, mask=feat_idx < input_features)

@torch.fx.wrap
def optimized_unsqueeze(input_tensor):
    batch_size, input_features = input_tensor.shape
    
    # Add dimension at position -2
    output_shape = (batch_size, 1, input_features)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    grid = (batch_size, input_features)
    
    # Launch kernel
    unsqueeze_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_features=input_features,
        BLOCK_SIZE_BATCH=1,
        BLOCK_SIZE_FEAT=1
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_unsqueeze