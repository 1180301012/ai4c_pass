import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel A: fp16 / bf16 — in_7 is 128KB (fits in L2).
# Focused autotune: BLOCK_K ∈ {64,128,256} × num_warps ∈ {4,8} = 6 configs.
# Each config gets ~4 warmup trials → more reliable selection.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K':  64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_K':  64}, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_K': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _fused_knet_fp16(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, in_5_ptr,
    in_6_ptr, in_7_ptr, in_8_ptr, in_9_ptr, in_10_ptr, in_11_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    j   = tl.arange(0, D)

    b0 = tl.load(in_0_ptr + j).to(tl.float32)
    w1 = tl.load(in_1_ptr + j).to(tl.float32)
    b2 = tl.load(in_2_ptr + j).to(tl.float32)
    w3 = tl.load(in_3_ptr + j).to(tl.float32)
    b4 = tl.load(in_4_ptr + j).to(tl.float32)
    w5 = tl.load(in_5_ptr + j).to(tl.float32)
    b6 = tl.load(in_6_ptr + j).to(tl.float32)

    # ── Linear: tiled K-loop ─────────────────────────────────────────────────
    acc = tl.zeros([D], dtype=tl.float32)
    for k_base in range(0, D, BLOCK_K):
        k    = k_base + tl.arange(0, BLOCK_K)
        x_k  = tl.load(in_8_ptr + row * D + k).to(tl.float32)
        w_jk = tl.load(in_7_ptr + j[:, None] * D + k[None, :]).to(tl.float32)
        acc += tl.sum(x_k[None, :] * w_jk, axis=1)
    linear_out = acc + b6

    # ── LayerNorm + sigmoid → tmp_11 ─────────────────────────────────────────
    mean_l = tl.sum(linear_out) / D
    x_l    = linear_out - mean_l
    tmp_9  = w3 * (x_l * tl.math.rsqrt(tl.sum(x_l * x_l) / D + EPS)) + b2
    tmp_11 = tl.sigmoid(tmp_9)

    # ── sigmoid(in_9) → tmp_10 ───────────────────────────────────────────────
    tmp_10 = tl.sigmoid(tl.load(in_9_ptr + row * D + j).to(tl.float32))

    # ── LayerNorm(in_11) → tmp_12 ────────────────────────────────────────────
    x11  = tl.load(in_11_ptr + row * D + j).to(tl.float32)
    x11c = x11 - tl.sum(x11) / D
    tmp_12 = w5 * (x11c * tl.math.rsqrt(tl.sum(x11c * x11c) / D + EPS)) + b4

    # ── LayerNorm(in_10) → tmp_13 ────────────────────────────────────────────
    x10  = tl.load(in_10_ptr + row * D + j).to(tl.float32)
    x10c = x10 - tl.sum(x10) / D
    tmp_13 = w1 * (x10c * tl.math.rsqrt(tl.sum(x10c * x10c) / D + EPS)) + b0

    tl.store(out_ptr + row * D + j,
             (tmp_11 * tmp_12 + tmp_10 * tmp_13).to(out_ptr.dtype.element_ty))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel B: fp32 — tiled loop over K with focused autotune configs.
# Avoids loading [256,256] fp32 (256KB) in one shot → manageable reg pressure.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K':  32}, num_warps=4),
        triton.Config({'BLOCK_K':  64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K':  32}, num_warps=8),
        triton.Config({'BLOCK_K':  64}, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_K':  32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K':  64}, num_warps=4, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _fused_knet_fp32(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, in_5_ptr,
    in_6_ptr, in_7_ptr, in_8_ptr, in_9_ptr, in_10_ptr, in_11_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    j   = tl.arange(0, D)

    b0 = tl.load(in_0_ptr + j).to(tl.float32)
    w1 = tl.load(in_1_ptr + j).to(tl.float32)
    b2 = tl.load(in_2_ptr + j).to(tl.float32)
    w3 = tl.load(in_3_ptr + j).to(tl.float32)
    b4 = tl.load(in_4_ptr + j).to(tl.float32)
    w5 = tl.load(in_5_ptr + j).to(tl.float32)
    b6 = tl.load(in_6_ptr + j).to(tl.float32)

    # ── Linear: tiled loop ───────────────────────────────────────────────────
    acc = tl.zeros([D], dtype=tl.float32)
    for k_base in range(0, D, BLOCK_K):
        k    = k_base + tl.arange(0, BLOCK_K)
        x_k  = tl.load(in_8_ptr + row * D + k).to(tl.float32)
        w_jk = tl.load(in_7_ptr + j[:, None] * D + k[None, :]).to(tl.float32)
        acc += tl.sum(x_k[None, :] * w_jk, axis=1)
    linear_out = acc + b6

    # ── LayerNorm + sigmoid → tmp_11 ─────────────────────────────────────────
    mean_l = tl.sum(linear_out) / D
    x_l    = linear_out - mean_l
    tmp_9  = w3 * (x_l * tl.math.rsqrt(tl.sum(x_l * x_l) / D + EPS)) + b2
    tmp_11 = tl.sigmoid(tmp_9)

    # ── sigmoid(in_9) → tmp_10 ───────────────────────────────────────────────
    tmp_10 = tl.sigmoid(tl.load(in_9_ptr + row * D + j).to(tl.float32))

    # ── LayerNorm(in_11) → tmp_12 ────────────────────────────────────────────
    x11  = tl.load(in_11_ptr + row * D + j).to(tl.float32)
    x11c = x11 - tl.sum(x11) / D
    tmp_12 = w5 * (x11c * tl.math.rsqrt(tl.sum(x11c * x11c) / D + EPS)) + b4

    # ── LayerNorm(in_10) → tmp_13 ────────────────────────────────────────────
    x10  = tl.load(in_10_ptr + row * D + j).to(tl.float32)
    x10c = x10 - tl.sum(x10) / D
    tmp_13 = w1 * (x10c * tl.math.rsqrt(tl.sum(x10c * x10c) / D + EPS)) + b0

    tl.store(out_ptr + row * D + j,
             (tmp_11 * tmp_12 + tmp_10 * tmp_13).to(out_ptr.dtype.element_ty))


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper: dispatch to dtype-appropriate kernel
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_knet(in_0, in_1, in_2, in_3, in_4, in_5,
               in_6, in_7, in_8, in_9, in_10, in_11):
    BATCH = in_8.shape[0]
    D     = 256
    EPS   = 1e-5
    out   = torch.empty(BATCH, 1, D, dtype=in_8.dtype, device=in_8.device)

    if in_8.element_size() <= 2:
        # fp16 / bf16: single-shot load, 2-config autotune (more reliable)
        _fused_knet_fp16[(BATCH,)](
            in_0, in_1, in_2, in_3, in_4, in_5,
            in_6, in_7, in_8, in_9, in_10, in_11,
            out, D=D, EPS=EPS,
        )
    else:
        # fp32: tiled loop, focused 6-config autotune
        _fused_knet_fp32[(BATCH,)](
            in_0, in_1, in_2, in_3, in_4, in_5,
            in_6, in_7, in_8, in_9, in_10, in_11,
            out, D=D, EPS=EPS,
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5,
            in_6, in_7, in_8, in_9, in_10, in_11):
    linear = torch.nn.functional.linear(in_8, in_7, in_6)
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


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5,
                     in_6, in_7, in_8, in_9, in_10, in_11):
    return (in_0, in_1, in_2, in_3, in_4, in_5,
            in_6, in_7, in_8, in_9, in_10, in_11)


def replacement_func():
    return fused_knet