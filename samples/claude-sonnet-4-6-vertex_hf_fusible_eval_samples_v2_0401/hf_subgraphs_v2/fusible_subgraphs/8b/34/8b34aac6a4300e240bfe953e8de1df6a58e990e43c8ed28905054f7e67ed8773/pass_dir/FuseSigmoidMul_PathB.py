import torch
import triton
import triton.language as tl


# Path B fusion: in_2 * sigmoid(conv2d)  →  one kernel (before interpolation)
def pattern(x, y):
    s = torch.sigmoid(x)
    r = y * s
    return r


def replacement_args(x, y):
    return (x, y)


@triton.jit
def sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x_val   = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y_val   = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result  = (y_val * tl.sigmoid(x_val)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def sigmoid_mul(x, y):
    out = torch.empty_like(y)
    n   = x.numel()
    if n < 131072:
        BS, NW = 1024, 4
    else:
        BS, NW = 4096, 8
    grid = (triton.cdiv(n, BS),)
    sigmoid_mul_kernel[grid](x, y, out, n, BLOCK_SIZE=BS, num_warps=NW)
    return out


def replacement_func():
    return sigmoid_mul