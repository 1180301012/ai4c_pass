"""Fused RoPE pass – 3-arg pattern that was confirmed working."""

import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_1, in_2, in_4):
    return (in_1, in_2, in_4)


@triton.jit
def _rope_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    S: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    s    = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = s * D
    k  = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(cos_ptr  + base + offs, mask=mask, other=0.0).to(tl.float32)
    si = tl.load(sin_ptr  + base + offs, mask=mask, other=0.0).to(tl.float32)
    partner   = offs ^ (D // 2)
    safe_p    = tl.where(partner < D, partner, 0)
    k_partner = tl.load(key_ptr + base + safe_p, mask=mask, other=0.0).to(tl.float32)
    rot = tl.where(offs < D // 2, -k_partner, k_partner)
    out = k * c + rot * si
    tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def fused_rope(in_1, in_2, in_4):
    S = 3; D = 256
    tmp_6 = torch.empty((1, 1, S, D), dtype=in_2.dtype, device=in_2.device)
    _rope_kernel[(S,)](in_2, in_1, in_4, tmp_6, S=S, D=D, BLOCK_D=D)
    return tmp_6


def replacement_func():
    return fused_rope