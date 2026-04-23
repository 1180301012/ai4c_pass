import torch
import triton
import triton.language as tl


# Try multiple patterns - first one that just matches torch.cat
# to see if the diagnostic matcher can give us useful info
def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    return (tmp_0,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def identity_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def identity_op(x):
    out = torch.empty_like(x)
    n = x.numel()
    identity_kernel[(n // 256 + 1,)](x, out, n, BLOCK_SIZE=256)
    return out


def replacement_func():
    return identity_op