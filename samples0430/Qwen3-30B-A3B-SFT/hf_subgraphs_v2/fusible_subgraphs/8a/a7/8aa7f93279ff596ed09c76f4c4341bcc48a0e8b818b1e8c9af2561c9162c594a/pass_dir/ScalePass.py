import torch
import triton
import triton.language as tl


# Pattern: matches in_1 * 0.1767766952966369
def pattern(in_1):
    return in_1 * 0.1767766952966369


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def scale_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Cast to fp32, scale, cast back to original dtype
    y = (x.to(tl.float32) * 0.1767766952966369).to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def triton_scale(in_1):
    n = in_1.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_1)
    scale_kernel[(num_blocks,)](in_1, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_scale