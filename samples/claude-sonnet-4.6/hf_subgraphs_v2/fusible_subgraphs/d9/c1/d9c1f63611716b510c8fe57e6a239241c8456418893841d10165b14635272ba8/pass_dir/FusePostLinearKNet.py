import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: everything after the linear layer (only tmp_17 is returned)
# ---------------------------------------------------------------------------
def pattern(linear, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0):
    tmp_9  = torch.nn.functional.layer_norm(linear, (256,), in_3, in_2, 1e-05)
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_12 = torch.nn.functional.layer_norm(in_11, (256,), in_5, in_4, 1e-05)
    tmp_13 = torch.nn.functional.layer_norm(in_10, (256,), in_1, in_0, 1e-05)
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(linear, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0):
    return (linear, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0)


# ---------------------------------------------------------------------------
# Kernel A:  sigmoid(LayerNorm(linear))  →  buf   [B, H]
#   grid = (B,)
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_a(
    linear_ptr,           # [B, 1, H]  contiguous → offset = b*H
    in3_ptr, in2_ptr,     # [H]  LN weight / bias
    buf_ptr,              # [B, H]  output
    H: tl.constexpr,
    EPS: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    b    = tl.program_id(0)
    feat = tl.arange(0, H)
    base = b * H

    x  = tl.load(linear_ptr + base + feat).to(tl.float32)
    mu = tl.sum(x, axis=0) / H
    x  = x - mu
    v  = tl.sum(x * x, axis=0) / H
    x  = x * tl.rsqrt(v + EPS)
    w  = tl.load(in3_ptr + feat).to(tl.float32)
    bv = tl.load(in2_ptr + feat).to(tl.float32)
    y  = tl.sigmoid(w * x + bv)

    if IS_FP16:
        tl.store(buf_ptr + base + feat, y.to(tl.float16))
    elif IS_BF16:
        tl.store(buf_ptr + base + feat, y.to(tl.bfloat16))
    else:
        tl.store(buf_ptr + base + feat, y)


# ---------------------------------------------------------------------------
# Kernel B:  sigmoid(in_9), LN(in_11), LN(in_10), combine  →  out  [B, 1, H]
#
#   out[b, 0, :] = buf[b] * LN(in_11)[b] + sigmoid(in_9)[b] * LN(in_10)[b]
#
#   grid = (B,)
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_b(
    buf_ptr,                       # [B, H]  sigmoid(LN(linear))
    in9_ptr,                       # [B, 1, H]
    in11_ptr, in5_ptr, in4_ptr,   # [B, H] + [H] + [H]  for LN(in_11)
    in10_ptr, in1_ptr, in0_ptr,   # [B, 1, H] + [H] + [H]  for LN(in_10)
    out_ptr,                       # [B, 1, H]
    H: tl.constexpr,
    EPS: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    b    = tl.program_id(0)
    feat = tl.arange(0, H)
    base = b * H

    # sigmoid(LN(linear))  —  already computed, just load
    tmp_11 = tl.load(buf_ptr + base + feat).to(tl.float32)

    # sigmoid(in_9)
    x9     = tl.load(in9_ptr + base + feat).to(tl.float32)
    tmp_10 = tl.sigmoid(x9)

    # LN(in_11)
    x11 = tl.load(in11_ptr + base + feat).to(tl.float32)
    mu  = tl.sum(x11, axis=0) / H
    x11 = x11 - mu
    v   = tl.sum(x11 * x11, axis=0) / H
    x11 = x11 * tl.rsqrt(v + EPS)
    w5  = tl.load(in5_ptr + feat).to(tl.float32)
    b4  = tl.load(in4_ptr + feat).to(tl.float32)
    tmp_12 = w5 * x11 + b4

    # LN(in_10)
    x10 = tl.load(in10_ptr + base + feat).to(tl.float32)
    mu  = tl.sum(x10, axis=0) / H
    x10 = x10 - mu
    v   = tl.sum(x10 * x10, axis=0) / H
    x10 = x10 * tl.rsqrt(v + EPS)
    w1  = tl.load(in1_ptr + feat).to(tl.float32)
    b0  = tl.load(in0_ptr + feat).to(tl.float32)
    tmp_13 = w1 * x10 + b0

    # combine:  sigmoid(LN(linear)) * LN(in_11)  +  sigmoid(in_9) * LN(in_10)
    out = tmp_11 * tmp_12 + tmp_10 * tmp_13

    if IS_FP16:
        tl.store(out_ptr + base + feat, out.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + base + feat, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + base + feat, out)


@torch.fx.wrap
def fused_postlinear_knet(linear, in3, in2, in9, in11, in5, in4, in10, in1, in0):
    B      = linear.shape[0]   # 300
    H      = linear.shape[2]   # 256
    dtype  = linear.dtype
    device = linear.device

    is_fp16 = (dtype == torch.float16)
    is_bf16 = (dtype == torch.bfloat16)

    # buf holds sigmoid(LN(linear)), shape [B, H]
    buf = torch.empty((B, H), dtype=dtype, device=device)

    _kernel_a[(B,)](
        linear, in3, in2, buf,
        H=H, EPS=1e-5,
        IS_FP16=is_fp16, IS_BF16=is_bf16,
        num_warps=4,
    )

    out = torch.empty((B, 1, H), dtype=dtype, device=device)

    _kernel_b[(B,)](
        buf, in9,
        in11, in5, in4,
        in10, in1, in0,
        out,
        H=H, EPS=1e-5,
        IS_FP16=is_fp16, IS_BF16=is_bf16,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_postlinear_knet