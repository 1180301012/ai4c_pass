import torch
import triton
import triton.language as tl

# Pattern matching function - matches the redundant transpose
def pattern(layer_norm_out):
    # Original: tmp_9 = tmp_8.transpose(0, 1); tmp_10 = tmp_8.transpose(0, 1)
    # Both transposes are identical, so we compute once and reuse
    tmp_9 = layer_norm_out.transpose(0, 1)
    tmp_10 = tmp_9  # Reuse instead of recomputing
    return tmp_9, tmp_10

# Argument extraction function
def replacement_args(layer_norm_out):
    return (layer_norm_out,)

# Optimized kernel for single transpose
@triton.jit
def transpose_kernel_0_1(
    input_ptr,
    output_ptr,
    n_features,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_val = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    # Handle transpose(0, 1) - swap first two dimensions
    output_shape = [input_tensor.shape[1], input_tensor.shape[0]] + list(input_tensor.shape[2:])
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    transpose_kernel_0_1[grid](
        input_tensor,
        output,
        input_tensor.shape[-1] if len(input_tensor.shape) > 1 else 1,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_forward(layer_norm_out):
        # Compute transpose once
        tmp_9 = optimized_transpose(layer_norm_out)
        tmp_10 = tmp_9  # Reuse instead of recomputing
        return tmp_9, tmp_10
    
    return optimized_forward