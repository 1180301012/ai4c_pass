import torch
import triton
import triton.language as tl

def pattern(a, b):
    a1 = a[..., :128]
    a2 = a[..., 128:]
    a2_neg = -a2
    cat = torch.cat([a2_neg, a1], dim=-1)
    return cat * b
def replacement_args(a, b):
    return (a, b)


@triton.jit
def triton_split_negate_concat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    cond = (offsets >= 128)
    a_vals = tl.where(cond, -a_vals, a_vals)

    out_vals = a_vals * b_vals

    tl.store(out_ptr + offsets, out_vals, mask=mask)


@torch.fx.wrap
def kernel_wrapper(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024

    out = torch.empty_like(a)

    triton_split_negate_concat_kernel[
        ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE),
    ](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
def replacement_func():
    return kernel_wrapper