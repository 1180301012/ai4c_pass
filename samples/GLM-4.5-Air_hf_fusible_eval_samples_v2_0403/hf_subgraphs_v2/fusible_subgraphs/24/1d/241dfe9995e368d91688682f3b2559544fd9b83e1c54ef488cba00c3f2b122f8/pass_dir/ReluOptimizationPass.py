import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Matches ReLU operation
    """
    return torch.nn.functional.relu(x)

def replacement_args(x):
    """Extract input tensor argument for the optimized kernel"""
    return (x,)

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized ReLU kernel
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    output = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    """
    Wrapper function for the optimized ReLU kernel
    """
    # Get tensor properties
    n_elements = x.numel()
    
    # Choose BLOCK_SIZE for optimal performance
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch the kernel
    relu_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized kernel wrapper function"""
    return optimized_relu