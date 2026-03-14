import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized sigmoid kernel using Triton.
    Each program processes BLOCK_SIZE elements in a flat 1D layout.
    """
    # Calculate starting position for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset array
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_val, mask=mask)


@torch.fx.wrap
def sigmoid_wrapper(x):
    """
    Optimized sigmoid using Triton kernel.
    x: input tensor of any shape
    """
    # Flatten to 1D for processing
    original_shape = x.shape
    x_flat = x.reshape(-1)
    n_elements = x_flat.numel()
    
    # Allocate output
    output = torch.empty_like(x_flat)
    
    # Choose block size - power of 2 for efficiency
    BLOCK_SIZE = 1024
    
    # Calculate grid - each program processes BLOCK_SIZE elements
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    sigmoid_kernel[grid](
        x_flat, output,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.reshape(original_shape)


def pattern(in_2):
    """
    Pattern: sigmoid activation
    """
    return in_2.sigmoid()


def replacement_args(in_2):
    return (in_2,)


def replacement_func():
    return sigmoid_wrapper