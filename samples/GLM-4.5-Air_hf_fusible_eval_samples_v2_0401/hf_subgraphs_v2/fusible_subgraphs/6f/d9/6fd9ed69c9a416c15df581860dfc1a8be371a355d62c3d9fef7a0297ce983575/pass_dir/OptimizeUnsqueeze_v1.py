import torch
import triton
import triton.language as tl

def pattern(tmp_6):
    """
    Match the final unsqueeze operation
    tmp_13 = tmp_6.unsqueeze(-2)
    tmp_6 = None  # exclude cleanup
    """
    tmp_13 = tmp_6.unsqueeze(-2)
    return tmp_13

def replacement_args(tmp_6):
    return (tmp_6,)

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Output data is the same (just with an extra dimension added)
    # The actual tensor data remains the same, only the stride changes
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze(input_tensor):
    """
    Optimized unsqueeze operation that preserves data layout while adding dimension
    """
    # The actual tensor data doesn't change, only the stride metadata
    # However, for optimization purposes we can create the output tensor efficiently
    
    # Get input shape and create output shape
    input_shape = input_tensor.shape
    output_shape = list(input_shape)
    output_shape.insert(-2, 1)  # Insert dimension at position -2
    
    # Create output tensor with the correct shape
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Copy data - since only dimension metadata changes, we can copy directly
    # Flatten both tensors to 1D for efficient copying
    flat_input = input_tensor.flatten()
    flat_output = output_tensor.flatten()
    
    # Copy data using Triton for performance
    if flat_input.numel() > 0:
        BLOCK_SIZE = 1024
        num_programs = (flat_input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        unsqueeze_kernel[(num_programs,)](
            flat_input,
            flat_output,
            flat_input.numel(),
            BLOCK_SIZE
        )
    
    # Set the stride to reflect the new dimension
    # This is crucial for the semantics of unsqueeze
    new_strides = list(output_tensor.stride())
    new_strides[-2] = 0  # The new dimension has stride 0
    output_tensor = output_tensor.as_strided(output_shape, new_strides)
    
    return output_tensor

def replacement_func():
    return optimized_unsqueeze