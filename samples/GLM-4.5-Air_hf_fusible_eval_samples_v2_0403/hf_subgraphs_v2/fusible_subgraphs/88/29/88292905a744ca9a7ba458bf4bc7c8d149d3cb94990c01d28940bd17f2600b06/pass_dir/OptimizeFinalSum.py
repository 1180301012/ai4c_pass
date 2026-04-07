import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    
    val = tl.load(input_ptr + pid)
    tl.store(output_ptr + pid, val)

@torch.fx.wrap
def optimized_contiguous(tmp_5):
    # Create contiguous copy efficiently
    n_elements = tmp_5.numel()
    output = torch.empty_like(tmp_5)
    
    contiguous_kernel[(n_elements, 1, 1)](
        input_ptr=tmp_5,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output

def replacement_func():
    return optimized_contiguous