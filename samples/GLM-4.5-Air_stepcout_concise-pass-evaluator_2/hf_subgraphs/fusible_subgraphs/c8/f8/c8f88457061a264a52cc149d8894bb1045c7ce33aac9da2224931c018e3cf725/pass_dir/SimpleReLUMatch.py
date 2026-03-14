import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Simple ReLU pattern matching"""
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    return tmp_2

def replacement_args(in_2):
    """Extract arguments for the ReLU kernel"""
    return (in_2,)

@triton.jit
def simple_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple ReLU kernel using Triton"""
    pid = tl.program_id(0)
    
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    relu_x = tl.maximum(x, 0.0)
    
    tl.store(out_ptr + offsets, relu_x, mask=mask)

@torch.fx.wrap
def simple_relu_triton(x):
    """Simple ReLU wrapper"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    simple_relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    """Return the simple ReLU function"""
    return simple_relu_triton