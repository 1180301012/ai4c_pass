"""
Shared Triton kernel + dispatch wrapper for the SE-attention pattern:
  conv2d(in_3, in_1, in_0) → hard-sigmoid → broadcast-scale in_2
Routes:
  "12"  →  hard_sigmoid(v) = clamp((v + 1) / 2,   0, 1)
  "36"  →  hard_sigmoid(v) = clamp((v + 3) * 1/6, 0, 1)
"""
import torch
import triton
import triton.language as tl


# ─── Triton kernel ──────────────────────────────────────────────────────────
# 1-D grid: (N * C_out,)  — one program per (batch, output-channel) pair.
# Each program:
#   1. Loads x[n, 0:C_in] and w[c_out, 0:C_in] ONCE (128-wide, masked).
#   2. Dot-product + bias  →  conv scalar output.
#   3. Hard-sigmoid activation.
#   4. Inner loop: scales every BLOCK_HW slice of in_2[n, c_out, :HW].
# Keeping the weight & input rows in registers/L1 avoids re-loading them
# for every HW block (unlike a 2-D grid that recomputes the GEMM each time).

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=4),
    ],
    key=['C_out', 'HW'],
)
@triton.jit
def _fused_se_kernel(
    x_ptr,      # [N, C_in, 1, 1]  – in_3 (contiguous; last 2 dims = 1)
    w_ptr,      # [C_out, C_in, 1, 1]
    bias_ptr,   # [C_out]
    in2_ptr,    # [N, C_out, H, W]
    out_ptr,    # [N, C_out, H, W]
    N, C_in, C_out, HW,
    add_val, inv_div,           # scalar activation constants (runtime)
    BLOCK_HW: tl.constexpr,
):
    pid   = tl.program_id(0)   # in [0, N * C_out)
    n     = pid // C_out
    c_out = pid % C_out

    # ── 1×1 conv: dot-product over C_in ≤ 128 (loaded once) ────────────────
    k_offs = tl.arange(0, 128)
    k_mask = k_offs < C_in
    xv = tl.load(x_ptr + n * C_in + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    wv = tl.load(w_ptr + c_out * C_in + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    acc = tl.sum(xv * wv, axis=0)
    acc = acc + tl.load(bias_ptr + c_out).to(tl.float32)

    # ── hard-sigmoid: clamp((acc + add_val) * inv_div, 0, 1) ────────────────
    attn = (acc + add_val) * inv_div
    attn = tl.minimum(tl.maximum(attn, 0.0), 1.0)

    # ── inner loop: scale BLOCK_HW slices of in_2[n, c_out, :HW] ────────────
    base = (n * C_out + c_out) * HW
    for hw0 in range(0, HW, BLOCK_HW):
        hw_offs = hw0 + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        x2  = tl.load(in2_ptr + base + hw_offs, mask=hw_mask, other=0.0)
        res = (x2.to(tl.float32) * attn).to(x2.dtype)
        tl.store(out_ptr + base + hw_offs, res, mask=hw_mask)


# ─── Shared dispatch wrapper ─────────────────────────────────────────────────
@torch.fx.wrap
def triton_fused_se_dispatch(in_0, in_1, in_2, in_3, route):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : feat   [N, C_out, H, W]
    in_3 : x_se   [N, C_in,  1, 1]
    route: "12"  → hard_sigmoid(v) = clamp((v+1)/2,   0,1)
           "36"  → hard_sigmoid(v) = clamp((v+3)/6,   0,1)
    """
    N     = in_3.shape[0]
    C_in  = in_3.shape[1]
    C_out = in_1.shape[0]
    HW    = in_2.shape[2] * in_2.shape[3]

    if route == "12":
        add_val = 1.0
        inv_div = 0.5                        # 1/2
    elif route == "36":
        add_val = 3.0
        inv_div = 0.16666666666666666        # 1/6
    else:
        add_val = 1.0
        inv_div = 0.5

    out = torch.empty_like(in_2)

    # 1-D grid: one program per (n, c_out) pair.
    _fused_se_kernel[(N * C_out,)](
        in_3, in_1, in_0, in_2, out,
        N, C_in, C_out, HW,
        add_val, inv_div,
    )
    return out