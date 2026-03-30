import torch
import triton
import triton.language as tl


def pattern(x, y):
    """
    Matches: sig = sigmoid(x); out = y * sig
    This pattern appears twice in the BiSeNetV2 graph:
      1. sigmoid(interp(in4)) * in3  at 64x64
      2. sigmoid(conv2d)      * in2  at 16x16
    """
    sig = torch.sigmoid(x)
    out = y * sig
    return out


def replacement_args(x, y):
    return (x, y)


@triton.jit
def sigmoid_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # tl.sigmoid requires fp32/fp64; cast x to fp32, compute, then cast back
    sig = tl.sigmoid(x.to(tl.float32)).to(x.dtype)

    # Fused sigmoid(x) * y in a single pass
    out = y * sig

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul(x, y):
    n_elements = x.numel()
    out = torch.empty_like(y)
    BLOCK_SIZE = 2048
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sigmoid_mul_kernel[(n_blocks,)](
        x, y, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_sigmoid_mul