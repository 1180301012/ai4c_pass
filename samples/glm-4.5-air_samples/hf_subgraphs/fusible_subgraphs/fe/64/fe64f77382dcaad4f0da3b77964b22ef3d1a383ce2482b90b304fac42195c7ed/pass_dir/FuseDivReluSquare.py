import torch
import triton
import triton.language as tl

@triton.jit
def fused_div_relu_square_kernel(x_ptr, out_ptr, n_elements, const_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Fused kernel: divide by constant -> ReLU -> square"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: x / const -> relu -> square
    divided = x / const_val
    relu_divided = tl.maximum(divided, 0.0)
    squared = relu_divided * relu_divided
    
    # Store result
    tl.store(out_ptr + offsets, squared, mask=mask)

@torch.fx.wrap
def fused_div_relu_square(x):
    """Fused Triton kernel for division + ReLU + squaring operations"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Handle different input shapes by computing grid size appropriately
    if len(x.shape) == 3:
        total_elements = x.shape[0] * x.shape[1] * x.shape[2]
    else:
        total_elements = n_elements
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel with the constant value
    const_val = 11.313708498984761
    fused_div_relu_square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        const_val=const_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Pattern matches: divide by 11.313708498984761 -> ReLU -> square"""
    # Create exactly the same computation as the original
    tmp_0 = x / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0) 
    tmp_2 = torch.square(tmp_1)
    return tmp_2

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_div_relu_square