import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0,)

@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    a: tl.constexpr, b: tl.constexpr, c: tl.constexpr, d: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert flattened index to 4D indices
    i = offsets // (b * c * d)
    rem = offsets % (b * c * d)
    j = rem // (c * d)
    rem = rem % (c * d)
    k = rem // d
    l = rem % d

    # Compute output index after transpose
    out_idx = i * (b * d * c) + j * (d * c) + l * c + k

    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_idx, x, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x):
    a, b, c, d = x.shape
    N = a * b * c * d
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    transpose_kernel[(num_programs,)](x, out, a, b, c, d, N, BLOCK_SIZE)
    return out

def replacement_func():
    return kernel_wrapper