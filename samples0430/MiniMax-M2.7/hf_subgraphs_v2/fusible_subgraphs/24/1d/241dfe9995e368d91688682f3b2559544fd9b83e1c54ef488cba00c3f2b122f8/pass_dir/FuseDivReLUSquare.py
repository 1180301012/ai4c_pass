import torch
import triton
import triton.language as tl

# Pattern matching function - matches the div -> relu -> square chain
def pattern(x):
    """
    Match the pattern: x / constant -> relu -> square
    The pattern is: in_0 / 11.313708498984761, then relu, then square
    """
    DIV_CONST = 11.313708498984761
    tmp = x / DIV_CONST
    tmp = torch.nn.functional.relu(tmp)
    result = torch.square(tmp)
    return result

def replacement_args(x):
    # Extract the input tensor - the pattern outputs only the final result
    return (x,)

@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    DIV_CONST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: square(relu(x / DIV_CONST))
    
    This fuses 3 operations into 1 kernel:
    1. Division by constant
    2. ReLU activation (max(0, x))
    3. Square (x * x)
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Operation 1: Division by constant
    x = x / DIV_CONST
    
    # Operation 2: ReLU - max(0, x)
    x = tl.where(x > 0, x, 0.0)
    
    # Operation 3: Square - x * x
    x = x * x
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_div_relu_square_kernel_wrapper(x):
    """
    Wrapper function to launch the fused kernel.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device
    output = torch.empty_like(x)
    
    # Launch kernel
    fused_div_relu_square_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        DIV_CONST=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_div_relu_square_kernel_wrapper