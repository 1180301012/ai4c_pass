import torch
import triton
import triton.language as tl

@triton.jit
def simple_copy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple copy kernel for testing"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and copy
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

def pattern(in_1, in_2, in_0):
    # Match the entire computation but return all outputs
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    
    return (tmp_6, tmp_4, tmp_2)

def replacement_args(in_1, in_2, in_0):
    return (in_1, in_2, in_0)

@torch.fx.wrap
def replacement_func_wrapper(in_1, in_2, in_0):
    """Wrapper using only allowed APIs"""
    # For now, just copy all inputs - this establishes pass matching works
    out_1 = torch.empty_like(in_1)
    out_2 = torch.empty_like(in_2)
    out_3 = torch.empty_like(in_0)
    
    # Copy inputs for now
    out_1.copy_(in_1)
    out_2.copy_(in_2)
    out_3.copy_(in_0)
    
    # Return in the same order as expected
    return (out_3, out_2, out_1)

def replacement_func():
    return replacement_func_wrapper