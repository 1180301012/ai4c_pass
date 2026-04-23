import torch
import triton
import triton.language as tl

def pattern(x):
    t = x / 1.6817928305074292
    out = t.transpose(-1, -2)
    return out

def replacement_args(x):
    return (x, 1.6817928305074292)

@triton.jit
def optimized_kernel(
    in_ptr, out_ptr,
    a, b, c, d,
    divisor,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    i = offsets // (b * d * c)
    rem = offsets % (b * d * c)
    j = rem // (d * c)
    rem = rem % (d * c)
    k = rem // c
    l = rem % c

    input_idx = i * (b * c * d) + j * (c * d) + l * d + k

    x_val = tl.load(in_ptr + input_idx, mask=mask)
    x_val = x_val.to(tl.float32) / divisor
    x_val = x_val.to(tl.float16)
    tl.store(out_ptr + offsets, x_val, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x, divisor):
    a, b, c, d = x.shape
    out = torch.empty((a, b, d, c), dtype=x.dtype, device=x.device)
    N = a * b * d * c
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    optimized_kernel[(num_programs,)](
        x, out,
        a, b, c, d,
        divisor,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    return kernel_wrapper