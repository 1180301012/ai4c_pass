import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    n_elements = x.numel()
    block_size = 1024
    grid_size = (n_elements + block_size - 1) // block_size
    
    out = torch.empty_like(x)
    add_kernel[grid_size]((x, y, out, n_elements, block_size))
    return out

class SimpleTritonAdd:
    @staticmethod
    def pattern(x, y):
        return x + y

    @staticmethod
    def replacement_args(x, y):
        return (x, y)

    @staticmethod
    def replacement_func():
        return triton_add

# Module-level functions for backward compatibility
def pattern(x, y):
    return SimpleTritonAdd.pattern(x, y)

def replacement_args(x, y):
    return SimpleTritonAdd.replacement_args(x, y)

def replacement_func():
    return SimpleTritonAdd.replacement_func()