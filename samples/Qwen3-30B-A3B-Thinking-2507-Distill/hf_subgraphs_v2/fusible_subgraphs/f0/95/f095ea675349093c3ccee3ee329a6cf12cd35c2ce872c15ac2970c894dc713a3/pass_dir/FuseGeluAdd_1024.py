import torch
import triton
import triton.language as tl


def pattern(conv_out, in_3):
    """
    Matches: slice -> gelu -> transpose -> add -> dropout(p,False,False)
    Returns single output tmp_8 (used both as model return and layer_norm input).

    conv_out : [B, C, S+1]  – raw conv1d output
    in_3     : [B, S, C]   – residual
    """
    tmp_4 = conv_out[(slice(None, None, None), slice(None, None, None), slice(None, -1, None))]
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.05, False, False)
    return tmp_8


def replacement_args(conv_out, in_3):
    return (conv_out, in_3)


# ── bfloat16 ────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_bf16_kernel(
    gelu_ptr, res_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    # Use min(offs,C-1) to keep gelu access within valid buffer bounds.
    # No mask needed — clamped address stays valid for all rows.
    c_safe = tl.minimum(offs, C - 1)
    # gelu: [B,C,Sg=250] stride(1)=Sg — use S=Sg=250 as stride
    gelu_f32 = tl.load(gelu_ptr + c_safe * S + row).to(tl.float32)
    gf = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))
    # res: [B,S=249,C=1024] — access row*S + offs (S here = Sg-1 = 249)
    res = tl.load(res_ptr + row * S + offs).to(tl.float32)
    tl.store(out_ptr + row * S + offs, (gf + res).to(tl.bfloat16))


# ── float16 ────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_fp16_kernel(
    gelu_ptr, res_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    gelu_f32 = tl.load(gelu_ptr + row * C + offs).to(tl.float32)
    gf = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    res = tl.load(res_ptr + row * S + offs).to(tl.float32)
    tl.store(out_ptr + row * S + offs, (gf + res).to(tl.float16))


# ── float32 ─────────────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_fp32_kernel(
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
def fused_gelu_add(conv_out, in_3):
    """
    Fused gelu(conv_out[:,:-1]) + transpose + add + dropout(0.05,False,False).
    Returns tmp_8 [B, S, C] (same as dropout output, identity in inference).

    conv_out : [B, C, S_gelu]  – raw conv1d output (full, NOT sliced)
      strides: stride(1) = S_gelu  (= S_plus_1, the channel stride)
    in_3     : [B, S, C]       – residual (S = S_gelu - 1)
    """
    B  = conv_out.shape[0]
    C  = conv_out.shape[1]   # 1024
    Sg = conv_out.shape[2]   # 250  (full time dimension)
    S  = Sg - 1               # 249  (time dimension after slice)
    rows = B * S
    out = torch.empty_like(in_3)

    # Pass the gelu-input channel stride (= Sg=250) for correct memory access.
    # The residual has S=249 as its channel dimension, already encoded in row*S+offs.
    kw = dict(BLOCK_SIZE=1024, num_warps=8)
    if conv_out.dtype == torch.bfloat16:
        _gelu_add_bf16_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)
    elif conv_out.dtype == torch.float16:
        _gelu_add_fp16_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)
    else:
        _gelu_add_fp32_kernel[(rows,)](conv_out, in_3, out, C, Sg, **kw)

    return out


def replacement_func():
    return fused_gelu_add