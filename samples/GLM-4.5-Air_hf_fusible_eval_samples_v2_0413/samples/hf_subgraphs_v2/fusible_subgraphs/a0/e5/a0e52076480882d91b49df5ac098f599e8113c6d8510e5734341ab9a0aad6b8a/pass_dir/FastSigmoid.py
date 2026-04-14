import torch
import triton
import triton.language as tl

# Pattern matching function for sigmoid optimization
def pattern(input_tensor):
    # Sigmoid activation using exact model variable names
    sigmoid_result = input_tensor.sigmoid()
    # Return the result
    return sigmoid_result

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for fast sigmoid approximation
@triton.jit
def fast_sigmoid_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Fast sigmoid approximation: 1 / (1 + exp(-x))
    # Use rational approximation for exp(-x) when x > 0, exp(x) when x < 0
    # For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
    
    # Split into positive and negative for better accuracy
    x_pos = tl.where(x >= 0, x, 0)
    x_neg = tl.where(x < 0, x, 0)
    
    # Approximation coefficients for exp(x) using polynomial
    # For exp(-x) where x >= 0 (limited range where approximation works well)
    exp_approx = 1.0 - 0.5 * x_pos + 0.208333 * x_pos * x_pos - 0.0625 * x_pos * x_pos * x_pos
    
    # For exp(x) where x < 0  
    exp_neg_approx = 1.0 + 0.5 * x_neg + 0.208333 * x_neg * x_neg + 0.0625 * x_neg * x_neg * x_neg
    
    # Combine results
    sigmoid_pos = 1.0 / (1.0 + exp_approx)
    sigmoid_neg = exp_neg_approx / (1.0 + exp_neg_approx)
    
    # Final sigmoid result
    sigmoid_result = tl.where(x >= 0, sigmoid_pos, sigmoid_neg)
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_result, mask=mask)

@torch.fx.wrap
def fast_sigmoid(input_tensor):
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fast_sigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fast_sigmoid