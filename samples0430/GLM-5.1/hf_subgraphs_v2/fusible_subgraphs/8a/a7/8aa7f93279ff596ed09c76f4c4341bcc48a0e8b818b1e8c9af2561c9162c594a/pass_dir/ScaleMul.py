import torch
import triton
import triton.language as tl


def pattern(in_1):
    return in_1 * 0.1767766952966369


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def scalar_mul_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out = x * scalar
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def scalar_mul_wrapper(in_1):
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    out = torch.empty_like(in_1)
    scalar_mul_kernel[(num_programs,)](
        input_ptr=in_1,
        output_ptr=out,
        n_elements=n_elements,
        scalar=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
    )
    return out


def replacement_func():
    return scalar_mul_wrapper