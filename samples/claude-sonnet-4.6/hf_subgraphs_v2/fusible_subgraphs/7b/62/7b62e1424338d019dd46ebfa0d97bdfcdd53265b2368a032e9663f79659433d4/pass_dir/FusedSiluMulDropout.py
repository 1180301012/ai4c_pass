import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Match only the gating mul - confirmed to anchor correctly
    out = in_0 * in_1
    return out


def replacement_args(in_0, in_1):
    # in_0 = silu(x) result, in_1 = gate; kernel just multiplies them
    return (in_0, in_1)


@triton.jit
def silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # x is already silu(input), y is gate; just multiply
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul_dropout(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_mul_kernel[grid](
        in_0, in_1, out, n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_silu_mul_dropout