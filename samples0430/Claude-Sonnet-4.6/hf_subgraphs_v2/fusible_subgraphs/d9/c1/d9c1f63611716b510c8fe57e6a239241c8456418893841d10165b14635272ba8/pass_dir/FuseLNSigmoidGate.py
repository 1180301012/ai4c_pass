import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Post-linear fused kernel: fuses 3×LayerNorm + 2×sigmoid + 2×mul + add.
# The linear (GEMM) is left to cuBLAS/cuDNN for maximum performance.
# Grid: (BATCH,) = (300,)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def _fused_post_linear_kernel(
    # LN params for in_10  (bias=in_0, weight=in_1)
    in_0_ptr, in_1_ptr,
    # LN params for linear output  (bias=in_2, weight=in_3)
    in_2_ptr, in_3_ptr,
    # LN params for in_11  (bias=in_4, weight=in_5)
    in_4_ptr, in_5_ptr,
    # activations
    in_9_ptr,      # input_gate   [BATCH, 1, D]
    in_10_ptr,     # input_out    [BATCH, 1, D]
    in_11_ptr,     # param_out    [BATCH, D]
    linear_ptr,    # linear(in_8, in_7, in_6) output  [BATCH, 1, D]
    # output
    out_ptr,       # [BATCH, 1, D]
    D: tl.constexpr,   # 256
    EPS: tl.constexpr, # 1e-5
):
    row = tl.program_id(0)
    j   = tl.arange(0, D)

    # ── load LN weights/biases (L2-cached — same for all rows) ───────────────
    b0 = tl.load(in_0_ptr + j).to(tl.float32)   # LN bias   for in_10
    w1 = tl.load(in_1_ptr + j).to(tl.float32)   # LN weight for in_10
    b2 = tl.load(in_2_ptr + j).to(tl.float32)   # LN bias   for linear
    w3 = tl.load(in_3_ptr + j).to(tl.float32)   # LN weight for linear
    b4 = tl.load(in_4_ptr + j).to(tl.float32)   # LN bias   for in_11
    w5 = tl.load(in_5_ptr + j).to(tl.float32)   # LN weight for in_11

    # ── Step 1: LayerNorm(linear[row], w=in_3, b=in_2) → tmp_9 → sigmoid → tmp_11
    x_lin  = tl.load(linear_ptr + row * D + j).to(tl.float32)
    mean_l = tl.sum(x_lin) / D
    x_l    = x_lin - mean_l
    var_l  = tl.sum(x_l * x_l) / D
    tmp_9  = w3 * (x_l * tl.math.rsqrt(var_l + EPS)) + b2
    tmp_11 = tl.sigmoid(tmp_9)

    # ── Step 2: sigmoid(in_9) → tmp_10 ──────────────────────────────────────
    x9     = tl.load(in_9_ptr + row * D + j).to(tl.float32)
    tmp_10 = tl.sigmoid(x9)

    # ── Step 3: LayerNorm(in_11[row], w=in_5, b=in_4) → tmp_12 ─────────────
    x11    = tl.load(in_11_ptr + row * D + j).to(tl.float32)
    mean11 = tl.sum(x11) / D
    x11c   = x11 - mean11
    var11  = tl.sum(x11c * x11c) / D
    tmp_12 = w5 * (x11c * tl.math.rsqrt(var11 + EPS)) + b4

    # ── Step 4: LayerNorm(in_10[row], w=in_1, b=in_0) → tmp_13 ─────────────
    x10    = tl.load(in_10_ptr + row * D + j).to(tl.float32)
    mean10 = tl.sum(x10) / D
    x10c   = x10 - mean10
    var10  = tl.sum(x10c * x10c) / D
    tmp_13 = w1 * (x10c * tl.math.rsqrt(var10 + EPS)) + b0

    # ── Step 5: tmp_11 * tmp_12 + tmp_10 * tmp_13 → output ──────────────────
    tmp_17 = tmp_11 * tmp_12 + tmp_10 * tmp_13
    tl.store(out_ptr + row * D + j, tmp_17.to(out_ptr.dtype.element_ty))


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (must be decorated with @torch.fx.wrap)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_post_linear(linear, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0):
    """
    Fused: 3×LayerNorm(256) + 2×sigmoid + 2×mul + add.
    linear : [BATCH, 1, D=256]  — output of torch.nn.functional.linear
    in_9   : [BATCH, 1, D=256]
    in_10  : [BATCH, 1, D=256]
    in_11  : [BATCH,    D=256]
    out    : [BATCH, 1, D=256]
    """
    BATCH = linear.shape[0]
    D     = 256
    EPS   = 1e-5

    out = torch.empty(BATCH, 1, D, dtype=linear.dtype, device=linear.device)

    _fused_post_linear_kernel[(BATCH,)](
        in_0, in_1, in_2, in_3, in_4, in_5,
        in_9, in_10, in_11, linear, out,
        D=D, EPS=EPS,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match (post-linear operations only; linear stays with cuBLAS)
# ─────────────────────────────────────────────────────────────────────────────
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


def replacement_func():
    return fused_post_linear