import torch
import triton
import triton.language as tl


def pattern(tmp_10, in_3):
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    return tmp_12


def replacement_args(tmp_10, in_3):
    return (tmp_10, in_3)


@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return triton_add