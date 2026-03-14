import torch
import triton
import triton.language as tl


# Pattern matching function - matches only the gelu pattern
def pattern(in_4):
    """
    Pattern: gelu(in_4)
    This matches the computation:
        tmp_5 = torch.nn.functional.gelu(in_4, approximate='none')
    Returns tmp_5 (the GELU output)
    """
    tmp_5 = torch.nn.functional.gelu(in_4, approximate='none')
    return tmp_5


# Argument extraction function
def replacement_args(in_4):
    return (in_4,)


# Triton kernel for GELU - using pure Triton with manual tanh implementation
@triton.jit
def gelu_kernel(
    in_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that performs: output = gelu(in)
    GELU formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    We implement tanh using: tanh(x) = 2 * sigmoid(2x) - 1
    """
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # GELU constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    # Compute x^3
    x3 = x * x * x
    
    # Compute inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = sqrt_2_over_pi * (x + coeff * x3)
    
    # Compute tanh using: tanh(x) = 2 * sigmoid(2x) - 1
    # sigmoid(x) = 1 / (1 + exp(-x))
    two_x = 2.0 * inner
    exp_neg_2x = tl.exp(-two_x)
    sigmoid_val = 1.0 / (1.0 + exp_neg_2x)
    tanh_val = 2.0 * sigmoid_val - 1.0
    
    # GELU output: 0.5 * x * (1 + tanh(inner))
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def gelu_kernel_wrapper(in_4):
    """
    Wrapper function that launches the Triton kernel for GELU.
    """
    # Flatten for 1D kernel
    orig_shape = in_4.shape
    in_flat = in_4.flatten()
    n_elements = in_flat.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out_flat = torch.empty_like(in_flat)
    
    # Launch kernel
    gelu_kernel[(num_programs,)](
        in_ptr=in_flat,
        output_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original shape
    return out_flat.reshape(orig_shape)


def replacement_func():
    """Returns the replacement function"""
    return gelu_kernel_wrapper