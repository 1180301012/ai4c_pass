import torch
import triton
import triton.language as tl


def pattern(in_0):
    return torch.sigmoid(in_0)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def sigmoid_kernel(
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
    # Cast to fp32 for numerical stability, then cast back
    sig = (1.0 / (1.0 + tl.exp(-x.to(tl.float32)))).to(x.dtype)
    tl.store(out_ptr + offsets, sig, mask=mask)


@torch.fx.wrap
def triton_sigmoid(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_kernel[grid](in_0, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_sigmoid