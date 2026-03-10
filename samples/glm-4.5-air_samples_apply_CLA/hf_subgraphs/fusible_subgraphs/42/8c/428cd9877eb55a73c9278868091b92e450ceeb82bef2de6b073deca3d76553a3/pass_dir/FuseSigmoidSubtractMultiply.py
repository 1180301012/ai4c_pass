import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    sigmoid_output = input_tensor.sigmoid()
    subtract_output = sigmoid_output - 0.25
    multiply_output = subtract_output * 3.141592653589793
    return sigmoid_output, multiply_output

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_sigmoid_subtract_multiply_kernel(
    input_ptr,
    output_ptr,
    sigmoid_output_ptr,  # For intermediate output if needed
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: sigmoid -> subtract -> multiply
    # We can optimize this using triton math functions
    sigmoid_x = tl.sigmoid(x)
    sigmoid_subtracted = sigmoid_x - 0.25
    result = sigmoid_subtracted * 3.141592653589793
    
    # Store both outputs (sigmoid and final result)
    tl.store(sigmoid_output_ptr + offsets, sigmoid_x, mask=mask)
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def fused_sigmoid_subtract_multiply_optimized_kernel(
    input_ptr,
    output_ptr,
    sigmoid_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations using triton math functions for better performance
    # tl.sigmoid is highly optimized for GPU
    sigmoid_x = tl.sigmoid(x)
    # Combine the arithmetic operations
    result = (sigmoid_x - 0.25) * 3.141592653589793
    
    # Store both outputs (sigmoid and final result) - sigmoid_x is needed as intermediate
    tl.store(sigmoid_output_ptr + offsets, sigmoid_x, mask=mask)
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_subtract_multiply_wrapper(input_tensor):
    n_elements = input_tensor.numel()
    
    # Create output tensors with same shape as input
    sigmoid_output = torch.empty_like(input_tensor)
    multiply_output = torch.empty_like(input_tensor)
    
    # Use optimal block size for GPU efficiency
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    fused_sigmoid_subtract_multiply_optimized_kernel[num_programs](
        input_tensor,
        multiply_output,
        sigmoid_output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sigmoid_output, multiply_output

def replacement_func():
    return fused_sigmoid_subtract_multiply_wrapper