"""
Fused optimization pass for:
  einsum('bchw,bchj->bhwj', query, key)  [B,C,H,W] x [B,C,H,J] -> [B,H,W,J]
  cat([energy, einsum], dim=-1)           -> [B,H,W,128]
  softmax(dim=-1)                         -> [B,H,W,128]
  out[..., :64]                           -> [B,H,W,64] (view)

Strategy:
  - One Triton program per (b, h) pair, processing all W positions together.
  - For a fixed (b, h):
      * query[b, c, h, :W] is contiguous (stride-1 in W dim) => coalesced loads
      * key[b, c, h, :J]   is contiguous (stride-1 in J dim) => coalesced loads
      * in_0[b, h, :W, :D] is a [W, D] contiguous block      => coalesced loads
  - Accumulate outer products over the C dimension to get the einsum result [W, J].
  - Compute softmax over the 128-wide concatenation in float32 for numerical accuracy.
  - Write only the final [B, H, W, 128] output — no intermediate tensors in global memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_einsum_cat_softmax_kernel(
    # Pointers
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    # Dimensions
    B, H, W,
    # Strides for in_0  [B, H, W, D]
    in0_sb, in0_sh, in0_sw,
    # Strides for in_1  [B, C, H, J]
    in1_sb, in1_sc, in1_sh,
    # Strides for in_2  [B, C, H, W]
    in2_sb, in2_sc, in2_sh,
    # Strides for out   [B, H, W, D2]
    out_sb, out_sh, out_sw,
    # Compile-time constants
    D: tl.constexpr,   # 64  (half-width; also equals C and J for this workload)
    D2: tl.constexpr,  # 128 (total output last-dim after cat)
):
    """
    Each program handles one (b, h) slice and computes all W output rows.
    Grid: B * H programs.
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    w_offs = tl.arange(0, D)   # [W = 64]
    j_offs = tl.arange(0, D)   # [J = 64]
    d_offs = tl.arange(0, D)   # [D = 64]

    # ------------------------------------------------------------------
    # 1. Compute einsum result:  acc[w, j] = sum_c( query[b,c,h,w] * key[b,c,h,j] )
    #    query: in_2[B, C, H, W],  key: in_1[B, C, H, J]
    # ------------------------------------------------------------------
    acc = tl.zeros([D, D], dtype=tl.float32)   # [W, J]

    q_bh = b * in2_sb + h * in2_sh   # base pointer to in_2[b, :, h, :]
    k_bh = b * in1_sb + h * in1_sh   # base pointer to in_1[b, :, h, :]

    for c in range(D):   # D == C == 64  (constexpr → compiler can pipeline)
        q_row = tl.load(in2_ptr + q_bh + c * in2_sc + w_offs).to(tl.float32)  # [W]
        k_row = tl.load(in1_ptr + k_bh + c * in1_sc + j_offs).to(tl.float32)  # [J]
        acc = acc + q_row[:, None] * k_row[None, :]   # outer-product accumulate [W,J]

    # ------------------------------------------------------------------
    # 2. Load in_0[b, h, :, :]  →  [W, D]  (contiguous)
    # ------------------------------------------------------------------
    in0_bh   = b * in0_sb + h * in0_sh
    in0_offs = w_offs[:, None] * in0_sw + d_offs[None, :]   # [W, D]
    in0_f    = tl.load(in0_ptr + in0_bh + in0_offs).to(tl.float32)  # [W, D]

    # ------------------------------------------------------------------
    # 3. Numerically-stable softmax over the 128-wide concatenated row
    # ------------------------------------------------------------------
    # Row-wise max across both halves
    row_max = tl.maximum(
        tl.max(in0_f, axis=1),   # [W]
        tl.max(acc,   axis=1),   # [W]
    )  # [W]

    exp_in0 = tl.exp(in0_f - row_max[:, None])   # [W, D]
    exp_acc = tl.exp(acc   - row_max[:, None])   # [W, J]

    row_sum = tl.sum(exp_in0, axis=1) + tl.sum(exp_acc, axis=1)  # [W]

    out_dtype = out_ptr.dtype.element_ty
    sm_in0 = (exp_in0 / row_sum[:, None]).to(out_dtype)   # [W, D]
    sm_acc = (exp_acc / row_sum[:, None]).to(out_dtype)   # [W, J]

    # ------------------------------------------------------------------
    # 4. Write output:  out[b, h, :, :D2]
    # ------------------------------------------------------------------
    out_bh = b * out_sb + h * out_sh

    # First D columns  (= energy part after softmax)
    out_offs_a = w_offs[:, None] * out_sw + d_offs[None, :]        # [W, D]
    tl.store(out_ptr + out_bh + out_offs_a, sm_in0)

    # Last D columns   (= einsum part after softmax)
    out_offs_b = w_offs[:, None] * out_sw + D + j_offs[None, :]   # [W, J]
    tl.store(out_ptr + out_bh + out_offs_b, sm_acc)


@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    """
    Drop-in replacement for:
        einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
        cat    = torch.cat([in_0, einsum], dim=-1)
        out    = torch.nn.functional.softmax(cat, dim=-1)
        return out, out[..., :64]

    in_0 : [B, H, W, D]   (energy)
    in_1 : [B, C, H, J]   (key)
    in_2 : [B, C, H, W]   (query)
    """
    B = in_2.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    D  = in_0.shape[-1]   # = 64
    J  = in_1.shape[-1]   # = 64
    D2 = D + J            # = 128

    out = torch.empty((B, H, W, D2), dtype=in_0.dtype, device=in_0.device)

    grid = (B * H,)

    fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2, out,
        B, H, W,
        # in_0 strides  [B, H, W, D]
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        # in_1 strides  [B, C, H, J]
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        # in_2 strides  [B, C, H, W]
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        # out strides   [B, H, W, D2]
        out.stride(0), out.stride(1), out.stride(2),
        D=D, D2=D2,
    )

    tmp_3 = out
    tmp_4 = out[..., :D]   # slice view — no copy
    return tmp_3, tmp_4


# ---------------------------------------------------------------------------
# Pattern / replacement API expected by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_einsum_cat_softmax