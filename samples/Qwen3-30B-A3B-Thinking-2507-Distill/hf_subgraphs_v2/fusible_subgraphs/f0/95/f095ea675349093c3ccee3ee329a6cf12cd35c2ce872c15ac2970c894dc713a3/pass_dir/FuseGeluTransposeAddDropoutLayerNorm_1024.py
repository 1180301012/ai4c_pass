import torch
import triton
import triton.language as tl


def pattern(tmp_4, in_3, in_1, in_0):
    """
    tmp_4 : [B, C, S] – gelu input (= conv_out[:, :, :-1])
    in_3  : [B, S, C] – residual
    in_1  : [C]       – layer-norm weight
    in_0  : [C]       – layer-norm bias
    """
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.05, False, False)
    tmp_10 = torch.nn.functional.layer_norm(tmp_8, (1024,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_10)


def replacement_args(tmp_4, in_3, in_1, in_0):
    return (tmp_4, in_3, in_1, in_0, "route_005")


# ── bfloat16 kernel ────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_ln_bf16_kernel(
    gelu_ptr, res_ptr, w_ptr, b_ptr,
    out_add_ptr, out_ln_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # gelu layout [B, C, S]: element [b,c,s] at b*C*S + c*S + s = row*S + c
    # res  layout [B, S, C]: element [b,s,c] at b*S*C + s*C + c = row*C + c
    gelu_f32 = tl.load(gelu_ptr + row * C + offs).to(tl.float32)
    res_f32  = tl.load(res_ptr  + row * S + offs).to(tl.float32)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_f32 = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    x_add = gelu_f32 + res_f32
    tl.store(out_add_ptr + row * S + offs, x_add.to(tl.bfloat16))

    mean    = tl.sum(x_add, axis=0) / BLOCK_SIZE
    diff    = x_add - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std
    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(out_ln_ptr + row * S + offs, (x_norm * w + b).to(tl.bfloat16))


# ── float16 kernel ─────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_ln_fp16_kernel(
    gelu_ptr, res_ptr, w_ptr, b_ptr,
    out_add_ptr, out_ln_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    gelu_f32 = tl.load(gelu_ptr + row * C + offs).to(tl.float32)
    res_f32  = tl.load(res_ptr  + row * S + offs).to(tl.float32)

    gelu_f32 = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    x_add = gelu_f32 + res_f32
    tl.store(out_add_ptr + row * S + offs, x_add.to(tl.float16))

    mean    = tl.sum(x_add, axis=0) / BLOCK_SIZE
    diff    = x_add - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std
    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(out_ln_ptr + row * S + offs, (x_norm * w + b).to(tl.float16))


# ── float32 kernel ─────────────────────────────────────────────────────────────
@triton.jit
def _gelu_add_ln_fp32_kernel(
    gelu_ptr, res_ptr, w_ptr, b_ptr,
    out_add_ptr, out_ln_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    gelu_f32 = tl.load(gelu_ptr + row * C + offs)
    res_f32  = tl.load(res_ptr  + row * S + offs)

    gelu_f32 = gelu_f32 * 0.5 * (1.0 + tl.math.erf(gelu_f32 * 0.7071067811865476))

    x_add = gelu_f32 + res_f32
    tl.store(out_add_ptr + row * S + offs, x_add)

    mean    = tl.sum(x_add, axis=0) / BLOCK_SIZE
    diff    = x_add - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std
    w = tl.load(w_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ln_ptr + row * S + offs, x_norm * w + b)


# ── shared dispatch wrapper (identical in both pass files) ────────────────────
@torch.fx.wrap
def _fused_dispatch_gelu_add_ln(arg0, arg1, arg2, arg3, route):
    # arg0=tmp_4 [B,C,S], arg1=in_3 [B,S,C], arg2=in_1 [C], arg3=in_0 [C]
    B, C, S = arg0.shape
    rows = B * S
    out_add = torch.empty_like(arg1)
    out_ln  = torch.empty_like(arg1)
    kw = dict(BLOCK_SIZE=1024, num_warps=8)
    if route == "route_005":
        _gelu_add_ln_bf16_kernel[(rows,)](arg0, arg1, arg2, arg3,
                                         out_add, out_ln,
                                         C, S, **kw)
    elif route == "route_01":
        _gelu_add_ln_fp16_kernel[(rows,)](arg0, arg1, arg2, arg3,
                                          out_add, out_ln,
                                          C, S, **kw)
    else:
        _gelu_add_ln_fp32_kernel[(rows,)](arg0, arg1, arg2, arg3,
                                          out_add, out_ln,
                                          C, S, **kw)
    return (out_add, out_ln)


def replacement_func():
    return _fused_dispatch_gelu_add_ln