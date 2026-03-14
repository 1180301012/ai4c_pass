import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern: division only"""
    tmp_0 = in_0 / 11.313708498984761
    return tmp_0

def replacement_args(in_0):
    """Extract the input tensor argument"""
    return (in_0,)

@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: division → ReLU → square in a single pass"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: divide → ReLU → square
    divided = x / divisor
    relu = tl.maximum(divided, 0.0)
    squared = relu * relu
    
    # Store result
    tl.store(out_ptr + offsets, squared, mask=mask)

@torch.fx.wrap
def simple_div(x):
    """Simple division implementation"""
    return x / 11.313708498984761

def replacement_func():
    """Return the simple function"""
    return simple_div