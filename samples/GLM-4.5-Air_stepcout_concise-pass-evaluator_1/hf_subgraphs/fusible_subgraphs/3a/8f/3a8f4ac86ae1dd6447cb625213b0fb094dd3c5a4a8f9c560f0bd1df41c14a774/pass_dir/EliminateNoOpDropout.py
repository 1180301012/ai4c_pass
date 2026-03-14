import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def eliminate_dropout(x):
    # Dropout with p=0.0 is a no-op, just compute ReLU with Triton
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def pattern(x):
    # ReLU -> Dropout (p=0.0) pattern
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    return tmp_1

def replacement_args(x):
    return (x,)

def replacement_func():
    return eliminate_dropout