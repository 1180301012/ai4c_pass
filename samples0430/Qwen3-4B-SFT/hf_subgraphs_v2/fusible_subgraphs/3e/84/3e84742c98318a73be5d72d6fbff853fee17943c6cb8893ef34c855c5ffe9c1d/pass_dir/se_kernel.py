"""
Shared Triton kernels for SE-block (Squeeze-and-Excitation) fusion.
Fuses: conv2d(1x1) + add + div + clamp + mul  in ONE pass.

Design:
  Grid: (B * Cout,)   -- one program per (b, c) pair
  Each program:
    (a) GEMM for scale[b,c] = clamp((GEMM_result + add_val)/div_val, 0, 1)
        using a loop over Cin with BLOCK_Ci tiles.
    (b) Broadcast-multiply in2[b, c, 0:HW] * scale[b,c]  using a loop over HW.
        BLOCK_HW elements handled per HW-step; unrolled when BLOCK_HW is constexpr.

No intermediate [B, Cout] buffer needed — scale written straight into
the broadcast-multiply output.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # {BLOCK_Ci=32, BLOCK_HW=256} is the proven best for large-batch cases
        triton.Config({'BLOCK_Ci': 32,  'BLOCK_HW':  256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_Ci': 64,  'BLOCK_HW':  256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_Ci': 64,  'BLOCK_HW':  256}, num_warps=8, num_stages=4),
        # Large HW covers small-batch (B=1/4) and medium-H cases
        triton.Config({'BLOCK_Ci': 32,  'BLOCK_HW':  512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_Ci': 64,  'BLOCK_HW':  512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_Ci': 64,  'BLOCK_HW':  512}, num_warps=8, num_stages=4),
        # Triton-native large blocks
        triton.Config({'BLOCK_Ci': 128, 'BLOCK_HW':  512}, num_warps=8, num_stages=4),
    ],
    key=['B', 'Cout', 'HW'],
)
@triton.jit
def _se_fused_kernel(
    in3_ptr,    # [B, Cin, 1, 1]  -- squeezed to [B, Cin]
    w_ptr,      # [Cout, Cin, 1, 1] -- squeezed to [Cout, Cin]
    bias_ptr,   # [Cout]
    in2_ptr,    # [B, Cout, H, W]
    out_ptr,    # [B, Cout, H, W]  (== in2 copy with per-channel scale applied)
    B, Cin, Cout, HW,
    add_val,    # 1.0  (SEOneC)
    div_val,    # 2.0  (SEOneC)
    BLOCK_Ci: tl.constexpr,   # tile size along Cin (power-of-2)
    BLOCK_HW: tl.constexpr,   # tile size along HW   (power-of-2)
):
    """
    Grid: (B * Cout,)
    Each program computes one (b, c) slice: out[b,c,:] = in2[b,c,:] * scale
    """
    pid = tl.program_id(0)
    b = pid // Cout
    c = pid % Cout

    # ------------------------------------------------------------------ #
    # Phase (a): GEMM  [B, Cin] x [Cin, Cout] + bias  ->  scale scalar   #
    # ------------------------------------------------------------------ #
    k_cur = tl.arange(0, BLOCK_Ci)
    acc   = tl.zeros([], dtype=tl.float32)
    for k_base in range(0, Cin, BLOCK_Ci):
        k_idx  = k_base + k_cur
        k_mask = k_idx < Cin
        x = tl.load(in3_ptr + b * Cin + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr   + c * Cin + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    conv_out = acc + tl.load(bias_ptr + c).to(tl.float32)
    tmp      = (conv_out + add_val) / div_val
    scale_f  = tl.minimum(tl.maximum(tmp, 0.0), 1.0)   # scalar float32

    # ------------------------------------------------------------------ #
    # Phase (b): in2[b, c, :] * scale_f  ->  out[b, c, :]                 #
    # ------------------------------------------------------------------ #
    hw_base = tl.arange(0, BLOCK_HW)
    for hw_start in range(0, HW, BLOCK_HW):
        hw_idx  = hw_start + hw_base
        hw_mask = hw_idx < HW
        data    = tl.load(in2_ptr + b * Cout * HW + c * HW + hw_idx,
                          mask=hw_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + b * Cout * HW + c * HW + hw_idx,
                 data * scale_f, mask=hw_mask)


# dtype helpers: maps eps_scale to the right add/div pair
_EPS_SCALE_MAP = {1.0: (1.0, 2.0), 3.0: (3.0, 6.0)}


def fused_se_forward(in0, in1, x, in2, eps_scale):
    """
    in0: bias   [Cout]
    in1: weight [Cout, Cin, 1, 1]
    x:   input  [B, Cin, 1, 1]
    in2: [B, Cout, H, W]
    eps_scale: 1.0 for one-scale (1/2) patterns, 3.0 for three-scale (3/6) patterns
    """
    B     = x.shape[0]
    Cin   = x.shape[1]
    Cout  = in1.shape[0]
    HW    = in2.shape[2] * in2.shape[3]

    out   = torch.empty_like(in2)

    ADD_V, DIV_V = _EPS_SCALE_MAP[eps_scale]

    # Autotune selects optimal BLOCK_Ci and BLOCK_HW per (B, Cout, HW) combination
    _se_fused_kernel[(B * Cout,)](
        x, in1, in0, in2, out,
        B, Cin, Cout, HW,
        ADD_V, DIV_V,
    )

    return out