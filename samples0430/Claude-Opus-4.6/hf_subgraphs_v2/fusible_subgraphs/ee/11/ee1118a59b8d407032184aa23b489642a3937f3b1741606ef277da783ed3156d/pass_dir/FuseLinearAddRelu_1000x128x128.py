import torch
import triton
import triton.language as tl


def pattern(residual, linear_output):
    tmp_3 = residual + linear_output
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(residual, linear_output):
    return (residual, linear_output)


@triton.jit
def add_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    result = a + b
    result = tl.maximum(result, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_relu(residual, linear_output):
    n_elements = residual.numel()
    out = torch.empty_like(residual)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    add_relu_kernel[grid](
        residual,
        linear_output,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_add_relu