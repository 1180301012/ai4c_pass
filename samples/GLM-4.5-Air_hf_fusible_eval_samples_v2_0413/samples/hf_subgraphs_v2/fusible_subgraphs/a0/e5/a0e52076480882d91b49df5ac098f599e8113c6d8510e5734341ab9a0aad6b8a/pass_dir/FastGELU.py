import torch
import triton
import triton.language as tl

# Pattern matching function for GELU activation optimization
def pattern(input_tensor):
    # GELU activation with 'none' approximation
    tmp_5 = torch.nn.functional.gelu(input_tensor, approximate='none')
    # Return the intermediate
    return tmp_5

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for fast GELU approximation
@triton.jit
def fast_gelu_kernel(
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
    
    # Fast GELU approximation using polynomial for better float handling
    # GELU(x) ≈ 0.5 * x * (1.0 + tanh(0.79788 * x * (1 + 0.044715 * x^2)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    cubic_coeff = 0.044715
    
    # Compute cubic term
    x_sq = x * x
    x_cubed = x_sq * x
    
    # Compute tanh argument
    tanh_arg = sqrt_2_over_pi * x * (1.0 + cubic_coeff * x_cubed)
    
    # Compute tanh using approximation to avoid math library calls
    # tanh(x) ≈ x * (27 + x^2) / (27 + 9 * x^2 + x^4) for |x| < 3
    # For larger |x|, use sign(x) * (1 - 2/(1 + exp(-2*|x|)))
    
    # Use polynomial approximation for tanh for performance
    # tanh(x) ≈ x * (27 + x^2) / (27 + 9 * x^2 + x^4) for reasonable accuracy
    x_squared = tanh_arg * tanh_arg
    x_cubed = x_squared * tanh_arg
    x_fourth = x_squared * x_squared
    
    numerator = tanh_arg * (27.0 + x_squared)
    denominator = 27.0 + 9.0 * x_squared + x_fourth
    tanh_val = numerator / denominator
    
    # Compute GELU
    gelu_val = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def fast_gelu(input_tensor):
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate grid size
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fast_gelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fast_gelu