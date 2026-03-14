import torch
import triton
import triton.language as tl

def pattern(arg1, arg2):
    # Handle the case where arguments might be passed in different order
    if isinstance(arg1, torch.Tensor) and len(arg1.shape) == 1:
        # arg1 is weight, arg2 is normalized_input
        weight = arg1
        normalized_input = arg2
    else:
        # arg1 is normalized_input, arg2 is weight (or scalar)
        normalized_input = arg1
        weight = arg2
        
    tmp_17 = weight * normalized_input
    return tmp_17

def replacement_args(arg1, arg2):
    return (arg1, arg2)

@triton.jit
def fused_weight_multiply_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    weight_size,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load the entire weight tensor (we assume it's small enough to load completely)
    weight_vals = tl.load(weight_ptr + tl.arange(0, weight_size), 
                        mask=tl.arange(0, weight_size) < weight_size, 
                        other=0.0)
    
    # Load input values for this block
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the hidden dimension index for each offset
    hidden_indices = offsets % weight_size
    
    # Broadcast weight and multiply
    broadcasted_weights = tl.load(weight_ptr + hidden_indices, mask=hidden_indices < weight_size, other=0.0)
    results = input_vals * broadcasted_weights
    
    # Store results
    tl.store(output_ptr + offsets, results, mask=mask)

@torch.fx.wrap
def fused_weight_multiply(arg1, arg2):
    # Handle the case where arguments might be passed in different order
    if isinstance(arg1, torch.Tensor) and len(arg1.shape) == 1:
        # arg1 is weight, arg2 is normalized_input
        weight = arg1
        normalized_input = arg2
    else:
        # arg1 is normalized_input, arg2 is weight (or scalar)
        normalized_input = arg1
        weight = arg2
    
    # If weight is a scalar, just do simple multiplication
    if isinstance(weight, (int, float)) or not torch.is_tensor(weight):
        return normalized_input * weight
    
    # Get input shapes
    weight_shape = weight.shape
    input_shape = normalized_input.shape
    
    # Validate shapes
    assert len(weight_shape) == 1, "Expected 1D weight"
    assert weight_shape[0] == 2048, "Expected weight size 2048"
    assert len(input_shape) == 3, "Expected 3D input"
    assert input_shape[2] == 2048, "Expected input hidden_dim 2048"
    
    batch_size, seq_len, hidden_dim = input_shape
    weight_size = weight_shape[0]
    total_elements = batch_size * seq_len * hidden_dim
    
    # Create output tensor
    outputs = torch.empty_like(normalized_input)
    
    # Set up launch grid
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_weight_multiply_kernel[(num_programs,)](
        weight,
        normalized_input,
        outputs,
        weight_size,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return outputs

def replacement_func():
    return fused_weight_multiply