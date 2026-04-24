import torch
import triton
import triton.language as tl


def pattern(conv_out, in_3):
    """
    Matches: slice -> gelu -> transpose -> add -> dropout(p=False,False) for p=0.1.
    Returns single output tmp_8.

    conv_out : [B, C, S+1]  – raw conv1d output
    in_3     : [B, S, C]   – residual
    """
    tmp_4 = conv_out[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False)
    return tmp_8


def replacement_args(conv_out, in_3):
    return (conv_out, in_3)


# ── bfloat16 ────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_bf16_v2_kernel(
    gelu_ptr, res_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    gelu_f32 = tl.load(gelu_ptr + row * C + offs).to(tl.float32)
    gf = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    res = tl.load(res_ptr + row * S + offs).to(tl.float32)
    tl.store(out_ptr + row * S + offs, (gf + res).to(tl.bfloat16))


# ── float16 ────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_fp16_v2_kernel(
    gelu_ptr, res_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # gelu_ptr: layout [B, C, S+1], Element [b,c,s] at row*S + c  (for B=1)
    # → access: gelu_ptr + offs*S + row
    # gelu_ptr: layout [B, C, Sg] where Sg=250 (full conv output, NOT the slice).
    # Channel stride = Sg.  Element [b,c,s] at b*C*Sg + c*Sg + s = row*Sg + c.
    # → access: gelu_ptr + offs*Sg + row
    # tl.minimum(offs, C-1) ensures valid (non-out-of-bounds) address for ALL rows.
    safe_offs = tl.minimum(offs, C - 1)
    gelu_f32 = tl.load(gelu_ptr + safe_offs * Sg + row,
                        mask=offs < C, other=0.0).to(tl.float32)
    gf = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    res = tl.load(res_ptr + row * S + offs).to(tl.float32)
    tl.store(out_ptr + row * S + offs, (gf + res).to(tl.float16))


# ── float32 ─────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_fp32_v2_kernel(
    gelu_ptr, res_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    gf = tl.load(gelu_ptr + row * C + offs)
    res = tl.load(res_ptr + row * S + offs)
    tl.store(out_ptr + row * S + offs, gf + res)


@torch.fx.wrap
def fused_gelu_add_v2(conv_out, in_3):
    """
    Fused gelu(conv_out[:,:-1]) + transpose + add + dropout(0.1,False,False).
    Returns tmp_8 [B, S, C].
    """
    B  = conv_out.shape[0]
    C  = conv_out.shape[1]   # 1024
    Sg = conv_out.shape[2]   # 250  (full time dimension)
    S  = Sg - 1               # 249  (time dimension after slice)
    rows = B * S
    out = torch.empty_like(in_3)

    kw = dict(BLOCK_SIZE=1024, num_warps=8)
    if conv_out.dtype == torch.bfloat16:
        _gelu_add_bf16_v2_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)
    elif conv_out.dtype == torch.float16:
        _gelu_add_fp16_v2_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)
    else:
        _gelu_add_fp32_v2_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)

    return out


def replacement_func():
    return fused_gelu_add_v2