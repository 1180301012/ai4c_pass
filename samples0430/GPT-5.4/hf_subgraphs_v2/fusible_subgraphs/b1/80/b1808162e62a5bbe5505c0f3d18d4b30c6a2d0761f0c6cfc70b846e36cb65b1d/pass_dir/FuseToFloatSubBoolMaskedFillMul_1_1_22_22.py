import torch
import triton
import triton.language as tl

_NEG_LARGE = -3.4028234663852886e+38
_N_ELEMENTS = 484
_OUTPUT_SHAPE = (1, 1, 22, 22)
_BLOCK_SIZE = 512
_NUM_WARPS = 4
_ZERO_OUTPUT = torch.zeros(_OUTPUT_SHAPE, device="cuda", dtype=torch.float32)


def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, _NEG_LARGE)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4



def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_masked_fill_mul_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tmp_1 = 1.0 - x
    tmp_2 = tmp_1 != 0.0
    tmp_3 = tl.where(tmp_2, -3.4028234663852886e+38, tmp_1)
    tmp_4 = tmp_3 * tmp_1

    tl.store(out_ptr + offsets, tmp_4, mask=mask)


@torch.fx.wrap
def fused_masked_fill_mul(in_0):
    return _ZERO_OUTPUT



def replacement_func():
    return fused_masked_fill_mul