import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_a = x[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_b = tmp_a.expand(1, 1, 8, 3, 256)
    tmp_c = tmp_b.reshape(1, 8, 3, 256)
    return tmp_c


def replacement_args(x):
    return (x,)


@triton.jit
def expand_kernel(
    src_ptr, dst_ptr,
    SD,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    sd_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sd_offs < SD
    vals = tl.load(src_ptr + sd_offs, mask=mask, other=0.0)
    for h in tl.static_range(0, H):
        tl.store(dst_ptr + h * SD + sd_offs, vals, mask=mask)


@torch.fx.wrap
def fast_expand(x):
    B, Hk, S, D = x.shape
    H_out = 8
    SD = S * D
    out = torch.empty(B, H_out, S, D, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 256
    grid = ((SD + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    expand_kernel[grid](x, out, SD, H_out, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return fast_expand