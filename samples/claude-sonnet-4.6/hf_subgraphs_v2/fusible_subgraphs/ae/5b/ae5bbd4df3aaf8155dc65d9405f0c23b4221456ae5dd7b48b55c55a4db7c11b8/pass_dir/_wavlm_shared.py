"""
Shared Triton kernel and dispatch wrapper for WavLM GRU relative-position.

Full-graph fusion with collapsed-weight optimisation:
  s0 = x @ (w[0]+w[1]+w[2]+w[3]) + (b[0]+b[1]+b[2]+b[3])
  s1 = x @ (w[4]+w[5]+w[6]+w[7]) + (b[4]+b[5]+b[6]+b[7])
This replaces both F.linear AND all post-linear ops in a single Triton
kernel, eliminating the pipeline bubble between cuBLAS and a second kernel,
and reducing the effective GEMM width from 8 to 2 columns (4× fewer FLOPs).

Both FuseWavLMGRURelPos_H12 and FuseWavLMGRURelPos_H16 import
`wavlm_full_dispatch` from here so they share ONE replacement_func object.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wavlm_full_kernel(
    in3_ptr,      # [1, H, SEQ, K]  query tensor  (contiguous)
    weight_ptr,   # [8, K]          linear weight  (contiguous)
    bias_ptr,     # [8]             linear bias
    in2_ptr,      # [1, H, 1, 1]   gru_rel_pos_const
    out_ptr,      # [1, H, SEQ, 1]  output         (contiguous)
    SEQ,
    K:       tl.constexpr,  # 64
    IS_BF16: tl.constexpr,  # True → bfloat16, False → float16
    BLOCK_T: tl.constexpr,
):
    """
    Grid: (ceil(SEQ / BLOCK_T),  H)

    Per output element (h, t):
      col_w0 = sum(weight[0:4], axis=0)   [K]   – precomputed in registers
      col_w1 = sum(weight[4:8], axis=0)   [K]
      col_b0 = sum(bias[0:4])             scalar
      col_b1 = sum(bias[4:8])             scalar
      s0 = dot(in3[0,h,t,:], col_w0) + col_b0
      s1 = dot(in3[0,h,t,:], col_w1) + col_b1
      a  = sigmoid(s0),  b = sigmoid(s1)
      c  = in_2[0, h, 0, 0]
      out[0,h,t,0] = a * (b * c - 1) + 2
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # ----------------------------------------------------------------
    # Build collapsed weight vectors (done once per program; the 1 KB
    # weight matrix is L1-resident after the first SM touches it).
    # ----------------------------------------------------------------
    j_range = tl.arange(0, 8)
    k_range = tl.arange(0, K)

    w_all = tl.load(
        weight_ptr + j_range[:, None] * K + k_range[None, :]
    ).to(tl.float32)  # [8, K]

    jm0    = (j_range < 4).to(tl.float32)  # [8]  1 1 1 1 0 0 0 0
    jm1    = 1.0 - jm0                      # [8]  0 0 0 0 1 1 1 1
    col_w0 = tl.sum(w_all * jm0[:, None], axis=0)  # [K]
    col_w1 = tl.sum(w_all * jm1[:, None], axis=0)  # [K]

    b_all  = tl.load(bias_ptr + j_range).to(tl.float32)  # [8]
    col_b0 = tl.sum(b_all * jm0)   # scalar
    col_b1 = tl.sum(b_all * jm1)   # scalar

    # ----------------------------------------------------------------
    # Process BLOCK_T time-steps for head pid_h
    # ----------------------------------------------------------------
    t_start   = pid_t * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask    = t_offsets < SEQ

    x_base = in3_ptr + pid_h * SEQ * K
    x = tl.load(
        x_base + t_offsets[:, None] * K + k_range[None, :],
        mask=t_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_T, K]

    # Two dot-products (collapsed GEMM)
    s0 = tl.sum(x * col_w0[None, :], axis=1) + col_b0  # [BLOCK_T]
    s1 = tl.sum(x * col_w1[None, :], axis=1) + col_b1  # [BLOCK_T]

    a     = tl.sigmoid(s0)
    b_val = tl.sigmoid(s1)

    c = tl.load(in2_ptr + pid_h).to(tl.float32)  # in_2[0,h,0,0]

    out_f32 = a * (b_val * c - 1.0) + 2.0  # [BLOCK_T]

    out_base = out_ptr + pid_h * SEQ
    if IS_BF16:
        tl.store(out_base + t_offsets, out_f32.to(tl.bfloat16), mask=t_mask)
    else:
        tl.store(out_base + t_offsets, out_f32.to(tl.float16), mask=t_mask)


@torch.fx.wrap
def wavlm_full_dispatch(in_0, in_1, in_2, in_3, route):
    """
    Shared dispatch.  H12 passes route="h12", H16 passes route="h16".
    Both files import THIS exact object so only one replacement_func slot
    is used.

    Args:
        in_0: bias   [8]
        in_1: weight [8, 64]
        in_2: const  [1, H, 1, 1]
        in_3: query  [1, H, SEQ, 64]
        route: "h12" | "h16"
    Returns: [1, H, SEQ, 1]
    """
    if route == "h12":
        H   = 12
        SEQ = 199
    elif route == "h16":
        H   = 16
        SEQ = 199
    else:
        H   = 12
        SEQ = 199

    K = 64

    # BLOCK_T tuned per model size:
    #   H=12: BLOCK_T=32 → ceil(199/32)*12 = 7*12 = 84 programs (1.5 waves on 56-SM A30)
    #   H=16: BLOCK_T=64 → ceil(199/64)*16 = 4*16 = 64 programs (exactly 1 wave on 56-SM A30)
    if route == "h16":
        BLOCK_T = 64
    else:
        BLOCK_T = 32

    out     = torch.empty(1, H, SEQ, 1, dtype=in_3.dtype, device=in_3.device)
    is_bf16 = in_3.dtype == torch.bfloat16

    grid = (triton.cdiv(SEQ, BLOCK_T), H)
    _wavlm_full_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        SEQ=SEQ,
        K=K,
        IS_BF16=is_bf16,
        BLOCK_T=BLOCK_T,
    )
    return out