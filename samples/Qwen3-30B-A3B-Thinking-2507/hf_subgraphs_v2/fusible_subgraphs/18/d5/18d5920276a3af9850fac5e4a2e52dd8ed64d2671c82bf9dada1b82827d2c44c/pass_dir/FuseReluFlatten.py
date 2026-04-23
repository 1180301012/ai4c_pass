import torch
import triton
import triton.language as tl

def pattern(in_0):
    relu_out = torch.nn.functional.relu(in_0, inplace=False)
    dropout_out = torch.nn.functional.dropout(relu_out, 0.0, False, False)
    flat_out = dropout_out.flatten(1, -1)
    return flat_out

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def triton_relu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    relu_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    return out

def replacement_func():
    return triton_relu