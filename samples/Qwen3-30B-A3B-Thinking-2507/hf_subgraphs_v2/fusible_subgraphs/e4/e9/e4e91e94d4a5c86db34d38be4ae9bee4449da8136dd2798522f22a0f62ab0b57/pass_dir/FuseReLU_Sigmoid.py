import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    relu_x = tl.maximum(x, 0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-relu_x))
    tl.store(out_ptr + offsets, sigmoid_x, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(x):
    n_elements = x.numel()
    BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    fused_relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_relu_sigmoid