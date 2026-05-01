import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.flatten(1, -1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fast_copy_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fast_flatten(in_0):
    B = in_0.shape[0]
    n_elements = in_0.numel()
    rest = n_elements // B
    out = torch.empty((B, rest), dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fast_copy_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fast_flatten