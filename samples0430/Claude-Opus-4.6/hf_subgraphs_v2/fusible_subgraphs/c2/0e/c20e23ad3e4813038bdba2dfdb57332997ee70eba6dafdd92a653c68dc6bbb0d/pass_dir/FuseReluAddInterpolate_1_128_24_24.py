import torch
import triton
import triton.language as tl


def pattern(a, b):
    return a + b


def replacement_args(a, b):
    return (a, b)


@triton.jit
def fused_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    out = a_val + b_val

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    out = torch.empty_like(a)

    fused_add_kernel[((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add