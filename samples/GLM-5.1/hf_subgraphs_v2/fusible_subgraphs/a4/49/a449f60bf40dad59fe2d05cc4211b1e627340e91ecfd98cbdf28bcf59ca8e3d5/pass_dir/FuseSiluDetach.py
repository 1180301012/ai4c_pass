import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0)
    tmp_3 = tmp_0.detach()
    return (tmp_3, tmp_0)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU = x * sigmoid(x)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_silu(in_0):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    tmp_0 = torch.empty_like(in_0)
    silu_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=tmp_0,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # tmp_3 is detach of tmp_0, same data, so return same tensor for both
    return (tmp_0, tmp_0)


def replacement_func():
    return triton_silu