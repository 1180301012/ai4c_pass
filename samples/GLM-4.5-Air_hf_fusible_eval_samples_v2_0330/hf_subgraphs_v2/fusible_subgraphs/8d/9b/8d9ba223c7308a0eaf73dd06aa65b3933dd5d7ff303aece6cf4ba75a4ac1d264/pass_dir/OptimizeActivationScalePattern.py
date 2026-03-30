import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_scale_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute optimized sigmoid * 16 in one step
    # sigmoid(x) * 16 = 1 / (1 + exp(-x)) * 16
    # We can compute this more efficiently by combining operations
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    sigmoid_result = 1.0 / (1.0 + exp_neg_x)
    final_result = 16.0 * sigmoid_result
    
    # Store result
    tl.store(output_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def optimized_sigmoid_scale(input):
    # Get total number of elements
    n_elements = input.numel()
    
    # Block size configuration
    BLOCK_SIZE = 1024
    
    # Grid configuration
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    sigmoid_scale_kernel[(num_programs,)](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(input_tensor):
    """Pattern: sigmoid activation followed by scaling by 16"""
    sigmoid_result = torch.sigmoid(input_tensor)
    scaled_result = 16 * sigmoid_result
    return scaled_result

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    return optimized_sigmoid_scale