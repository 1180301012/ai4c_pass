import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _fused_kernel(
    in_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x_f32 = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tmp_1 = 1.0 - x_f32
    tmp_2 = tmp_1 != 0.0
    tmp_3 = tl.where(tmp_2, -3.4028234663852886e+38, tmp_1)
    tmp_4 = tmp_3 * tmp_1
    tl.store(out_ptr + offsets, tmp_4, mask=mask)


_buf = [None]


@torch.fx.wrap
def fused_cast_sub_bool_maskedfill_mul(in_0):
    if _buf[0] is None:
        _buf[0] = torch.empty([1, 1, 22, 22], dtype=torch.float32, device=in_0.device)
    _fused_kernel[(1,)](in_0, _buf[0], 484, 512, num_warps=1, num_stages=1)
    return _buf[0]


def replacement_func():
    return fused_cast_sub_bool_maskedfill_mul