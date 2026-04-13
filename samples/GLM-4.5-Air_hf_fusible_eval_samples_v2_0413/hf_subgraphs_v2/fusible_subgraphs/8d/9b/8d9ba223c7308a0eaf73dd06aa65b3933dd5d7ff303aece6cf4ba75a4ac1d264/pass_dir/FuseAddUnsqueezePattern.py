import torch
import triton
import triton.language as tl

# Pattern matching function for addition with unsqueeze pattern
def pattern(base_tensor, bias_tensor):
    """
    Match pattern: add operations with unsqueeze broadcasting
    This appears multiple times in the target computation, e.g.:
    tmp_12 = in_2 + tmp_11  # where tmp_11 = tmp_10.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15  # where tmp_15 = in_3.unsqueeze(1).unsqueeze(0)
    """
    # Match the pattern: unsqueeze + add with specific broadcasting
    tmp_14 = base_tensor.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    result = bias_tensor + tmp_15
    return result

# Argument extraction function
def replacement_args(base_tensor, bias_tensor):
    return (base_tensor, bias_tensor)

# Optimized kernel using Triton
@triton.jit
def add_unsqueeze_kernel(
    base_ptr,
    bias_ptr,
    output_ptr,
    base_n_elements,
    bias_n_elements,
    output_n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_n_elements
    
    # Load bias data (this should be broadcastable)
    bias_data = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Load base tensor data with proper broadcasting simulation
    # The unsqueeze patterns create broadcasting that we need to handle efficiently
    input_data = tl.load(base_ptr + (offsets % base_n_elements), mask=mask, other=0.0)
    
    # Perform addition with optimized broadcasting
    output_data = input_data + bias_data
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)

@torch.fx.wrap
def optimized_add_unsqueeze(base_tensor, bias_tensor):
    """
    Optimized version of add operations with unsqueeze broadcasting
    """
    # Create output tensor with correct shape
    output_shape = bias_tensor.shape
    output = torch.empty(output_shape, dtype=bias_tensor.dtype, device=bias_tensor.device)
    
    # Launch Triton kernel for larger tensors
    n_elements = output.numel()
    if n_elements > 2048:  # Use Triton for larger tensors
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        add_unsqueeze_kernel[(num_programs,)](
            base_ptr=base_tensor,
            bias_ptr=bias_tensor,
            output_ptr=output,
            base_n_elements=base_tensor.numel(),
            bias_n_elements=bias_tensor.numel(),
            output_n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For small tensors, use regular operations
        result = base_tensor.unsqueeze(1).unsqueeze(0)
        output = bias_tensor + result
    
    return output

# Replacement function
def replacement_func():
    return optimized_add_unsqueeze