import torch
import operator
import triton
import triton.language as tl


def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(x):
    return (x,)


@triton.jit
def avgpool_flatten_kernel(
    x_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per channel. Computes global average pooling.
    out[c] = mean(x[c, 0..HW-1])
    """
    c = tl.program_id(0)
    base = c * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    avg = tl.sum(x, axis=0) / HW
    tl.store(out_ptr + c, avg.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_relu_avgpool(x):
    # Fuses adaptive_avg_pool2d(x, 1) + flatten(1, -1) into one op.
    # mean(dim=(-2,-1)) is equivalent to global average pooling,
    # and directly produces the flattened [1, C] shape, saving one dispatch.
    return x.mean(dim=(-2, -1))


def replacement_func():
    return fused_relu_avgpool