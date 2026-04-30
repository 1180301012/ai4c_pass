import torch
import triton
import triton.language as tl


def pattern(x):
    return x.reshape(1, 512, 16, 16)


def replacement_args(x):
    return (x,)


@triton.jit
def identity_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def triton_reshape(x):
    output = torch.empty(1, 512, 16, 16, dtype=x.dtype, device=x.device)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    identity_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output


def replacement_func():
    return triton_reshape