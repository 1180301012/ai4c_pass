import torch
import triton
import triton.language as tl


def pattern(x, y):
    """
    Matches: sig = sigmoid(x); out = y * sig
    After FuseSigmoidMulAdd consumes the 64x64 (sigmoid+mul+add) chain,
    this matches the remaining 16x16 branch: sigmoid(conv2d) * in2
    """
    sig = torch.sigmoid(x)
    out = y * sig
    return out


def replacement_args(x, y):
    return (x, y)


@triton.jit
def sigmoid_mul_b2_kernel(
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

    # Cast to fp32 for sigmoid (Triton requires fp32/fp64), cast result back
    sig = tl.sigmoid(x.to(tl.float32)).to(x.dtype)
    out = y * sig

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul_b2(x, y):
    n_elements = x.numel()
    out = torch.empty_like(y)
    BLOCK_SIZE = 2048
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sigmoid_mul_b2_kernel[(n_blocks,)](
        x, y, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_sigmoid_mul_b2