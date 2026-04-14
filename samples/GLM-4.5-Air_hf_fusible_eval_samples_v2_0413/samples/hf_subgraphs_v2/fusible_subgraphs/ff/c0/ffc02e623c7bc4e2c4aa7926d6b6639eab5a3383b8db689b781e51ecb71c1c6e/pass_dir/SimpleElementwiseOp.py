import torch
import triton
import triton.language as tl

def pattern(x):
    result = x * 2.0
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def simple_elementwise_kernel(x_ptr, output_ptr, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    
    val = tl.load(x_ptr + pid)
    result = val * 2.0
    tl.store(output_ptr + pid, result)

@torch.fx.wrap
def simple_elementwise(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    num_programs = (n_elements + 1023) // 1024
    
    simple_elementwise_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements
    )
    
    return output

def replacement_func():
    return simple_elementwise