import torch
import triton
import triton.language as tl

def pattern(x):
    y = x.view(1, -1)
    z = y.repeat(2, 1)
    return z

def replacement_args(x):
    return (x,)

@triton.jit
def repeat_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)
    tl.store(output_ptr + N + offsets, x, mask=mask)

@torch.fx.wrap
def triton_repeat(x):
    N = x.numel()
    out = torch.empty((2, N), dtype=x.dtype, device=x.device)
    input_ptr = x.data_ptr()
    output_ptr = out.data_ptr()
    BLOCK_SIZE = 128
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    repeat_kernel[(num_blocks,)](input_ptr, output_ptr, N, BLOCK_SIZE)
    return out

def replacement_func():
    return triton_repeat