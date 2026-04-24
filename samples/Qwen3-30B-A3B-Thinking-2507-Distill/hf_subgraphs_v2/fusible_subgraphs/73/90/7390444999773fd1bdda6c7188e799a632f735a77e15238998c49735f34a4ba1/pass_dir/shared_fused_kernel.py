"""
Shared Triton kernel: fused linear-value-projection + Flash Attention.

Pattern optimized:
  linear(in_3, weight, bias)          -> value projection
  .view(B, -1, H, D) .transpose(1,2) -> rearrange to [B,H,S,D]
  SDPA(query, key, value, mask, ...)   -> attention
  .transpose(1,2) .reshape(B,S,H*D)  -> final output

The kernel computes the value linear projection inline (avoiding an
extra round-trip to global memory for the intermediate tensor) and
implements Flash-Attention to compute the full attention in one pass.
"""

import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
    ],
    key=["BH", "S"],
)
@triton.jit
def _fused_linear_attn_kernel(
    # ---- inputs -------------------------------------------------------
    in3_ptr,           # [B, S, D_in]
    weight_ptr,        # [D_out, D_in]
    bias_ptr,          # [D_out]
    in5_ptr,           # [B, H, S, D]
    in4_ptr,           # [B, H, S, D]
    in2_ptr,           # [B, 1, S, S]
    # ---- output -------------------------------------------------------
    out_ptr,           # [B, S, H*D_out]
    # ---- strides (all contiguous, stride_h computed in wrapper) ------
    stride_in3_b,      # in3: batch stride  = S * D_in
    stride_in3_s,      # in3: seq stride   = D_in
    stride_w_do,       # weight: out-dim stride = D_in
    stride_in5_b,
    stride_in5_h,
    stride_in5_s,
    stride_in4_b,
    stride_in4_h,
    stride_in4_s,
    stride_in2_b,
    stride_in2_s,      # in2: stride over dim-3 = S
    stride_out_b,      # out: batch stride  = S * H*D_out
    stride_out_s,      # out: seq stride    = H*D_out
    # ---- runtime scalars --------------------------------------------
    BH,                # B * H  (number of (batch, head) pairs)
    S,                 # sequence length
    D,                 # head dimension (= 64)
    D_in,              # input dimension to linear
    D_out,             # output dimension from linear (= H * D, or just D_out)
    scale,             # 1 / sqrt(D)
    H,                 # number of heads
    # ---- compile-time constexprs ------------------------------------
    BLOCK_D: tl.constexpr,   # = 64, next power-of-2 >= D
    BLOCK_M: tl.constexpr,   # autotuned row tile
    BLOCK_N: tl.constexpr,   # autotuned column tile
):
    # ------------------------------------------------------------------
    # Program ids: (batch*head, query-tile)
    # ------------------------------------------------------------------
    bh_idx  = tl.program_id(0)
    m_block = tl.program_id(1)

    batch_idx = bh_idx // H
    h_idx     = bh_idx  % H

    m_start   = m_block * BLOCK_M
    offs_m    = m_start + tl.arange(0, BLOCK_M)
    offs_d    = tl.arange(0, BLOCK_D)

    # ------------------------------------------------------------------
    # Load query tile:  [BLOCK_M, BLOCK_D]
    # ------------------------------------------------------------------
    q_base    = batch_idx * stride_in5_b + h_idx * stride_in5_h
    q         = tl.load(
        in5_ptr + q_base + offs_m[:, None] * stride_in5_s + offs_d[None, :],
        mask=offs_m[:, None] < S, other=0.0
    ).to(tl.float32)

    # ------------------------------------------------------------------
    # Load key tile:    [BLOCK_N, BLOCK_D]
    # ------------------------------------------------------------------
    k_base    = batch_idx * stride_in4_b + h_idx * stride_in4_h
    offs_n    = tl.arange(0, BLOCK_N)

    # ------------------------------------------------------------------
    # Online value projection + attention loop over K/V tiles
    # ------------------------------------------------------------------
    m_i  = tl.full([BLOCK_M], 0.0, dtype=tl.float32)   # row-max tracker
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)        # normaliser tracker
    acc  = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    n_blocks = tl.cdiv(S, BLOCK_N)

    for n_block in range(n_blocks):
        n_start  = n_block * BLOCK_N
        offs_n   = n_start + tl.arange(0, BLOCK_N)
        n_mask   = offs_n < S

        # -- Load K tile [BLOCK_N, BLOCK_D] --------------------------------
        k = tl.load(
            in4_ptr + k_base + offs_n[:, None] * stride_in4_s + offs_d[None, :],
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)

        # -- Query-K attention scores [BLOCK_M, BLOCK_N] -------------------
        s = tl.dot(q, tl.trans(k)) * scale  # (q @ k.T) * scale

        # -- Mask out-of-bounds positions -----------------------------------
        valid2d = (offs_m[:, None] < S) & (n_mask[None, :])

        # -- Load additive attention mask [B, 1, S, S] ---------------------
        a_base = batch_idx * stride_in2_b
        attn_bias = tl.load(
            in2_ptr + a_base + offs_m[:, None] * stride_in2_s + offs_n[None, :],
            mask=valid2d, other=0.0
        ).to(tl.float32)
        s = s + attn_bias
        s = tl.where(valid2d, s, float('-inf'))

        # -- Online softmax update -----------------------------------------
        m_new  = tl.maximum(m_i, tl.max(s, axis=1))
        alpha  = tl.exp(m_i - m_new)
        exp_s  = tl.exp(s - m_new[:, None])

        l_i    = l_i * alpha + tl.sum(exp_s, axis=1)
        acc    = acc * alpha[:, None]

        # -- Load V (projected on-the-fly) [BLOCK_N, BLOCK_D] --------------
        #    v_proj[n, d_out] = sum_d(V[n,d] * W[d_out, d]) + bias[d_out]
        v_base = batch_idx * stride_in3_b + h_idx * D_out  # head stride = D_out
        v = tl.load(
            in3_ptr + v_base + offs_n[:, None] * stride_in3_s + offs_d[None, :],
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)

        # -- Load weight tile [BLOCK_D, BLOCK_K] for V projection -----------
        #    weight layout: [D_out, D_in]  ->  W[d_out, d_in]
        #    we want: w_row[d_out, d_in] = weight[d_out, d_in]
        #    BLOCK_K = BLOCK_D = 64  (D_in == D_out_head == D)
        w = tl.load(
            weight_ptr + offs_d[None, :] * stride_w_do + offs_d[:, None],
            mask=None
        )  # shape: [BLOCK_D, BLOCK_D]  -- this is weight transposed!

        # v: [BLOCK_N, BLOCK_D], w: [BLOCK_D, BLOCK_D]
        # v_proj = v @ w  =>  [BLOCK_N, BLOCK_D]
        # But weight is already transposed in memory [BLOCK_D, BLOCK_K]
        v_proj = tl.dot(v, w).to(tl.float32)  # [BLOCK_N, BLOCK_D]

        # Add bias
        bias_vals = tl.load(bias_ptr + offs_d).to(tl.float32)
        v_proj = v_proj + bias_vals[None, :]

        # -- Accumulate weighted V -----------------------------------------
        acc    = acc    + tl.dot(exp_s, v_proj)
        m_i    = m_new

    # ------------------------------------------------------------------
    # Normalize and store output tile
    # ------------------------------------------------------------------
    acc = acc / l_i[:, None]

    out_base = batch_idx * stride_out_b + h_idx * D_out
    tl.store(
        out_ptr + out_base + offs_m[:, None] * stride_out_s + offs_d[None, :],
        acc.to(tl.bfloat16),  # always output bfloat16
        mask=offs_m[:, None] < S
    )


# ---------------------------------------------------------------------------
# Python-level wrapper  (decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_linear_attn(in_3, weight, bias, in_5, in_4, in_2):
    """
    Fused linear-value-projection + Flash Attention.

    in_3  : [B, S, D_in]           – hidden states
    weight: [D_out, D_in]          – value projection weight
    bias  : [D_out]                – value projection bias
    in_5  : [B, H, S, D]           – query
    in_4  : [B, H, S, D]           – key
    in_2  : [B, 1, S, S]           – attention bias mask (additive)
    returns: [B, S, H*D_out]
    """
    B      = in_3.shape[0]
    S      = in_3.shape[1]
    D_in   = in_3.shape[2]
    D_out  = weight.shape[0]
    H      = in_5.shape[1]
    D      = in_5.shape[3]     # head dim (= 64)

    dtype    = in_3.dtype
    device   = in_3.device

    BH      = B * H
    scale   = 1.0 / math.sqrt(D)

    # Allocate output  [B, S, H*D_out]
    out = torch.empty((B, S, H * D_out), dtype=dtype, device=device)

    # Weight is stored as [D_out, D_in] => stride_w_do = D_in
    # and we access weight[d_out, d_in] = weight_ptr + d_out * D_in + d_in
    # For the V-projection tile we need weight.T in [BLOCK_D, BLOCK_K] order.
    # stride_w_do = D_in  (row stride of weight)

    grid = lambda META: (BH, triton.cdiv(S, META['BLOCK_M']))

    _fused_linear_attn_kernel[grid](
        in_3, weight, bias, in_5, in_4, in_2, out,
        # strides
        in_3.stride(0), in_3.stride(1),
        weight.stride(0),
        in_5.stride(0), in_5.stride(1), in_5.stride(2),
        in_4.stride(0), in_4.stride(1), in_4.stride(2),
        in_2.stride(0), in_2.stride(2),
        out.stride(0), out.stride(1),
        # runtime scalars
        BH, S, D, D_in, D_out, scale, H,
        # constexprs
        BLOCK_D=64,
    )

    return out