import torch
import triton
import triton.language as tl

def pattern(tmp_4, in_0):
    tmp_3 = tmp_4
    tmp_3 += in_0
    tmp_5 = torch.nn.functional.relu(tmp_3, inplace=True)
    return tmp_5

def replacement_args(tmp_4, in_0):
    return (tmp_4, in_0)

@triton.jit
def add_relu_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Add and apply ReLU in one operation
    sum_val = in_0 + in_1
    relu_val = tl.maximum(sum_val, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_val, mask=mask)

@torch.fx.wrap
def add_relu_kernel_wrapper(in_0, in_1):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return add_relu_kernel_wrapper