import torch
import triton
import triton.language as tl


def pattern(in_3, in_4):
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(in_3, in_4):
    return (in_3, in_4)


# Fixed block size – avoids autotune Python overhead on every call.
# For the range of tensor sizes we encounter (2M – 145M elements),
# a BLOCK_SIZE of 16384 with 8 warps gives the best bandwidth utilisation.
_BLOCK_SIZE = 16384
_NUM_WARPS  = 8


@triton.jit
def _fused_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def fused_add_dropout2d_eval(in_3, in_4):
    out = torch.empty_like(in_4)
    n   = in_4.numel()
    _fused_add_kernel[((n + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)](
        in_4, in_3, out, n,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return out


def replacement_func():
    return fused_add_dropout2d_eval