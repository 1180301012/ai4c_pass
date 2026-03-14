import torch
import triton
import triton.language as tl


# Pattern matching function - matches the GELU approximation computation
def pattern(in_0):
    """
    Match the GELU approximation pattern:
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7
    """
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for GELU approximation
@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    # Constants from the computation
    const_0 = 0.5
    const_1 = 0.044715
    const_2 = 0.7978845608028654
    const_3 = 1.0
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner expression: x + 0.044715 * x^3
    inner = x + const_1 * x_cubed
    
    # Multiply by scaling constant: 0.7978845608028654 * inner
    scaled = const_2 * inner
    
    # Compute tanh using exp formula: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # Here scaled = 0.7978845608028654 * (x + 0.044715 * x^3)
    # We need tanh(scaled), so we compute 2*scaled and use the formula
    two_scaled = 2.0 * scaled
    exp_2x = tl.exp(two_scaled)
    tanh_result = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Compute 1 + tanh
    one_plus_tanh = const_3 + tanh_result
    
    # Final: 0.5 * x * (1 + tanh(...))
    result = const_0 * x * one_plus_tanh
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_gelu(x):
    """Triton kernel wrapper for fused GELU approximation."""
    # Handle multi-dimensional tensor by flattening
    original_shape = x.shape
    x_flat = x.view(-1)
    n_elements = x_flat.numel()
    
    # Choose block size - use 1024 for good occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    output = torch.empty_like(x_flat)
    
    # Launch kernel
    gelu_kernel[(num_programs,)](
        x_ptr=x_flat,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original shape
    return output.view(original_shape)


# Replacement function - returns the wrapper function
def replacement_func():
    return triton_gelu