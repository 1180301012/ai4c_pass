import torch
import triton
import triton.language as tl

def pattern(in_0, in_2, in_3):
    t1 = in_3 + in_0
    t2 = t1 + in_2
    t3 = torch.nn.functional.relu(t2)
    return t3

def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)

@triton.jit
def add_relu_kernel(
    in0_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(in3_ptr + offsets, mask=mask, other=0.0)
    
    s = a + b + c
    out = tl.maximum(s, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_relu(in_0, in_2, in_3):
    n = in_0.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_0)
    add_relu_kernel[(num_blocks,)](
        in0_ptr=in_0,
        in2_ptr=in_2,
        in3_ptr=in_3,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_add_relu