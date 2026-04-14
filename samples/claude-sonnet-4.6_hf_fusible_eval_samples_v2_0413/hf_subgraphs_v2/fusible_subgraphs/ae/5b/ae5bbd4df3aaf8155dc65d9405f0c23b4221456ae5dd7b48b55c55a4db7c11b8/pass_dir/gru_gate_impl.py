# Shared Triton kernel and dispatch function for GRU gate fusion.
# Both FuseGRUGate_H12 and FuseGRUGate_H16 import from here so that
# replacement_func() returns the SAME function object across all passes,
# satisfying the output_pass_replacement_func_limit constraint.
#
# Design: 2D grid (H, S) — one program per output element.
# Each program computes the full fused pipeline for one (head, seq) pair:
#   linear → view(2,4).sum(-1) → sigmoid → gate arithmetic
# Eliminates CUBLAS overhead + 6+ elementwise kernel launches.

import torch
import triton
import triton.language as tl


@triton.jit
def _gru_gate_kernel(
    in3_ptr,       # [1, H, S, K]  input tensor (query)
    weight_ptr,    # [8, K]        linear weight
    bias_ptr,      # [8]           linear bias
    in2_ptr,       # [1, H, 1, 1]  gru_rel_pos_const (contiguous → offset = h)
    out_ptr,       # [1, H, S, 1]  output
    S,             # seq length (199)
    K: tl.constexpr,      # hidden dim = 64
    BLOCK_S: tl.constexpr,  # sequences per program (tunable)
):
    """
    2D grid: axis-0 = head h, axis-1 = sequence block pid_s.
    Each program handles BLOCK_S consecutive sequences for one head.

    Weight-loading amortization: weight [8, K] and bias [8] are loaded
    ONCE per program and reused across all BLOCK_S outputs, reducing
    L2 traffic by BLOCK_S× compared to BLOCK_S=1.

    Fused computation per sequence s in [s_base, s_base+BLOCK_S):
      x    = in3[0, h, s, :]                  [K]
      y[i] = dot(x, weight[i]) + bias[i]      i in 0..7
      sum0 = y[0]+y[1]+y[2]+y[3]
      sum1 = y[4]+y[5]+y[6]+y[7]
      g0   = sigmoid(sum0)
      g1   = sigmoid(sum1)
      out  = g0 * (g1 * in2[0,h,0,0] - 1.0) + 2.0
    """
    h     = tl.program_id(0)
    pid_s = tl.program_id(1)
    s_base = pid_s * BLOCK_S

    k_off  = tl.arange(0, K)       # [K]
    s_off  = tl.arange(0, BLOCK_S) # [BLOCK_S]
    s_abs  = s_base + s_off         # absolute sequence indices [BLOCK_S]
    s_mask = s_abs < S              # boundary mask [BLOCK_S]

    # ------------------------------------------------------------------
    # Load weight rows [K] — loaded ONCE, reused for all BLOCK_S seqs
    # ------------------------------------------------------------------
    w0 = tl.load(weight_ptr + 0 * K + k_off).to(tl.float32)
    w1 = tl.load(weight_ptr + 1 * K + k_off).to(tl.float32)
    w2 = tl.load(weight_ptr + 2 * K + k_off).to(tl.float32)
    w3 = tl.load(weight_ptr + 3 * K + k_off).to(tl.float32)
    w4 = tl.load(weight_ptr + 4 * K + k_off).to(tl.float32)
    w5 = tl.load(weight_ptr + 5 * K + k_off).to(tl.float32)
    w6 = tl.load(weight_ptr + 6 * K + k_off).to(tl.float32)
    w7 = tl.load(weight_ptr + 7 * K + k_off).to(tl.float32)

    # Load bias [8]
    b0 = tl.load(bias_ptr + 0).to(tl.float32)
    b1 = tl.load(bias_ptr + 1).to(tl.float32)
    b2 = tl.load(bias_ptr + 2).to(tl.float32)
    b3 = tl.load(bias_ptr + 3).to(tl.float32)
    b4 = tl.load(bias_ptr + 4).to(tl.float32)
    b5 = tl.load(bias_ptr + 5).to(tl.float32)
    b6 = tl.load(bias_ptr + 6).to(tl.float32)
    b7 = tl.load(bias_ptr + 7).to(tl.float32)

    # Load in2[0, h, 0, 0]
    in2_val = tl.load(in2_ptr + h).to(tl.float32)

    # ------------------------------------------------------------------
    # Load BLOCK_S input rows:  x[BLOCK_S, K]
    # Address layout: in3[0, h, s, k] = in3_ptr + (h*S + s)*K + k
    # ------------------------------------------------------------------
    x = tl.load(
        in3_ptr + (h * S + s_abs[:, None]) * K + k_off[None, :],
        mask=s_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_S, K]

    # ------------------------------------------------------------------
    # 8 dot products per sequence, broadcast weight across BLOCK_S dim
    # y_i[b] = sum_k(x[b,k] * w_i[k]) + b_i,  b in [0,BLOCK_S)
    # ------------------------------------------------------------------
    y0 = tl.sum(x * w0[None, :], axis=1) + b0  # [BLOCK_S]
    y1 = tl.sum(x * w1[None, :], axis=1) + b1
    y2 = tl.sum(x * w2[None, :], axis=1) + b2
    y3 = tl.sum(x * w3[None, :], axis=1) + b3
    y4 = tl.sum(x * w4[None, :], axis=1) + b4
    y5 = tl.sum(x * w5[None, :], axis=1) + b5
    y6 = tl.sum(x * w6[None, :], axis=1) + b6
    y7 = tl.sum(x * w7[None, :], axis=1) + b7

    # Group-of-4 sums [BLOCK_S]
    sum0 = y0 + y1 + y2 + y3
    sum1 = y4 + y5 + y6 + y7

    # Sigmoid gates [BLOCK_S]
    g0 = tl.sigmoid(sum0)
    g1 = tl.sigmoid(sum1)

    # Gate combination [BLOCK_S]
    out_val = g0 * (g1 * in2_val - 1.0) + 2.0

    # Store [BLOCK_S] with boundary mask
    tl.store(out_ptr + h * S + s_abs, out_val, mask=s_mask)


@torch.fx.wrap
def fused_gru_gate(in_0, in_1, in_2, in_3, route):
    """
    Shared dispatch wrapper used by both H=12 and H=16 passes.
    H is inferred from in_3 at runtime; 'route' disambiguates passes.

    Args:
        in_0: bias   [8]
        in_1: weight [8, 64]
        in_2: const  [1, H, 1, 1]
        in_3: query  [1, H, S, 64]
        route: "h12" or "h16"
    Returns:
        out: [1, H, S, 1]
    """
    H = in_3.shape[1]
    S = in_3.shape[2]
    K = 64

    out = torch.empty((1, H, S, 1), dtype=in_3.dtype, device=in_3.device)

    # 2D grid: (H, ceil(S / BLOCK_S))
    # BLOCK_S=8: each program handles 8 sequences, loading weight only once
    # for 8 outputs → 8× reduction in L2 weight bandwidth per output.
    # num_warps=2 → 64 threads matches K=64 element width exactly.
    BLOCK_S = 8
    grid = (H, triton.cdiv(S, BLOCK_S))
    _gru_gate_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        S, K, BLOCK_S,
        num_warps=2,
    )
    return out