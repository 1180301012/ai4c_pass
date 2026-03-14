import torch
import triton
import triton.language as tl

def pattern(weight, normalized_input):
    tmp_17 = weight * normalized_input
    return tmp_17

def replacement_args(weight, normalized_input):
    return (weight, normalized_input)

@triton.jit
def optimized_weight_multiply_kernel(
    weight_ptr,
    normalized_ptr, 
    output_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load weight and normalized input values
    weight_val = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    normalized_val = tl.load(normalized_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    result = weight_val * normalized_val
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_weight_multiply(weight, normalized_input):
    # Check if we're dealing with scalar weight
    if not torch.is_tensor(weight) or weight.numel() == 1:
        # If weight is scalar, just return element-wise multiplication
        return weight * normalized_input
    
    # Get input shapes
    weight_shape = weight.shape
    input_shape = normalized_input.shape
    
    # For simple 1D weight * 3D input broadcasting case
    if len(weight_shape) == 1 and len(input_shape) == 3:
        batch_size, seq_len, hidden_dim = input_shape
        weight_size = weight_shape[0]
        assert weight_size == hidden_dim, f"Expected weight size {hidden_dim}, got {weight_size}"
        
        # Flatten the input for more efficient processing
        total_elements = batch_size * seq_len * hidden_dim
        flattened_input = normalized_input.reshape(-1)
        flattened_output = torch.empty(total_elements, dtype=normalized_input.dtype, device=normalized_input.device)
        
        # Set up launch grid
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_weight_multiply_kernel[(num_programs,)](
            weight,
            flattened_input,
            flattened_output,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return flattened_output.reshape(input_shape)
    else:
        # Fall back to simple multiplication for other cases
        return weight * normalized_input

def replacement_func():
    return optimized_weight_multiply