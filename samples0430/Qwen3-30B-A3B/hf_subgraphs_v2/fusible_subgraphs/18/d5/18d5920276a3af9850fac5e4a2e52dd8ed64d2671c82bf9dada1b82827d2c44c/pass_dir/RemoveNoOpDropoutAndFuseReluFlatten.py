import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_relu_and_flatten(in_0):
    n = in_0.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_out = torch.empty_like(in_0)
    relu_kernel[(num_blocks,)](
        in_0, relu_out, n, BLOCK_SIZE
    )
    return relu_out.view(in_0.shape[0], -1)

def pattern(in_0):
    tmp0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp1 = torch.nn.functional.dropout(tmp0, 0.0, False, False)
    tmp2 = tmp1.flatten(1, -1)
    return tmp2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return optimized_relu_and_flatten