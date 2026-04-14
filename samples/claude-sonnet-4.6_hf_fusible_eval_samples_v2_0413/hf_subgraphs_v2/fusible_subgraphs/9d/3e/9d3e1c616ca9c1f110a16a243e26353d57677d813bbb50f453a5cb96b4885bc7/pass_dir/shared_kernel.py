"""
Shared Triton kernel + dispatch wrapper for Conv1×1 + BN(inference) + Residual Add.

Three route strings cover the three model variants:
  "A"  start96_end99_0   :  conv(in_6, in_4),  BN(in_0,in_1,in_3,in_2),  +in_5
  "B"  start116_end119_1 :  conv(in_5, in_4),  BN(in_0,in_1,in_3,in_2),  +in_6
  "C"  start23_end26_7   :  conv(in_5, in_0),  BN(in_1,in_2,in_4,in_3),  +in_6
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "Cin", "Cout"],
)
@triton.jit
def _conv1x1_bn_add(
    x_ptr,       # conv input  [N, Cin, H, W]  (NCHW)
    w_ptr,       # conv weight [Cout, Cin]      (1x1 → 2-D)
    mean_ptr,    # BN running_mean [Cout]
    var_ptr,     # BN running_var  [Cout]
    bn_w_ptr,    # BN weight  [Cout]
    bn_b_ptr,    # BN bias    [Cout]
    res_ptr,     # residual   [N, Cout, H, W]  (NCHW)
    out_ptr,     # output     [N, Cout, H, W]  (NCHW)
    M,           # N * H * W
    Cin,
    Cout,
    HW,          # H * W
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (ceil(M/BLOCK_M), ceil(Cout/BLOCK_N))

    NCHW address helpers:
      x  [m, k]:   (b*Cin + k)*HW + hw   where b=m//HW, hw=m%HW
      W  [n, k]:   n*Cin + k
      res[m, n]:   (b*Cout + n)*HW + hw
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_ids = m_start + tl.arange(0, BLOCK_M)
    n_ids = n_start + tl.arange(0, BLOCK_N)

    hw_ids = m_ids % HW
    b_ids  = m_ids // HW

    m_mask = m_ids < M
    n_mask = n_ids < Cout

    # ---- GEMM (Conv 1x1) ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(Cin, BLOCK_K)):
        k_ids  = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_ids < Cin

        # X tile [BM, BK]: stride-HW in the K direction (NCHW)
        x_ptrs = (b_ids[:, None] * Cin + k_ids[None, :]) * HW + hw_ids[:, None]
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # W tile [BN, BK]: contiguous row-major
        w_ptrs = n_ids[:, None] * Cin + k_ids[None, :]
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    # ---- BN epilogue (per output-channel) ----
    mean = tl.load(mean_ptr + n_ids, mask=n_mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + n_ids, mask=n_mask, other=0.0).to(tl.float32)
    bn_w = tl.load(bn_w_ptr + n_ids, mask=n_mask, other=0.0).to(tl.float32)
    bn_b = tl.load(bn_b_ptr + n_ids, mask=n_mask, other=0.0).to(tl.float32)

    scale = bn_w / tl.sqrt(var + eps)
    shift = bn_b - mean * scale
    acc   = acc * scale[None, :] + shift[None, :]

    # ---- Residual add + store ----
    res_ptrs = (b_ids[:, None] * Cout + n_ids[None, :]) * HW + hw_ids[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]

    res_tile = tl.load(res_ptr + res_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    acc += res_tile

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + res_ptrs, acc.to(out_dtype), mask=out_mask)


# ---------------------------------------------------------------------------
# Helper: run the kernel for given logical args
# ---------------------------------------------------------------------------

def _run(conv_input, conv_weight, mean, var, bn_w, bn_b, residual):
    """conv_weight shape: [Cout, Cin, 1, 1]  (1x1 conv, no bias)"""
    N, Cin, H, W = conv_input.shape
    Cout = conv_weight.shape[0]
    M  = N * H * W
    HW = H * W

    out = torch.empty((N, Cout, H, W),
                      dtype=conv_input.dtype,
                      device=conv_input.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(Cout, meta["BLOCK_N"]))

    _conv1x1_bn_add[grid](
        conv_input, conv_weight,
        mean, var, bn_w, bn_b,
        residual, out,
        M, Cin, Cout, HW,
        1e-5,
    )
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (ALL pass files must return THIS object)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def dispatch_conv_bn_add(in_0, in_1, in_2, in_3, in_4, in_5, in_6, route):
    """
    Route "A": conv(in_6,in_4), BN(mean=in_0,var=in_1,w=in_3,b=in_2), +in_5
    Route "B": conv(in_5,in_4), BN(mean=in_0,var=in_1,w=in_3,b=in_2), +in_6
    Route "C": conv(in_5,in_0), BN(mean=in_1,var=in_2,w=in_4,b=in_3), +in_6
    Returns a single tensor (match_output=False, no tuple wrapping needed).
    """
    if route == "A":
        return _run(in_6, in_4, in_0, in_1, in_3, in_2, in_5)
    elif route == "B":
        return _run(in_5, in_4, in_0, in_1, in_3, in_2, in_6)
    else:   # "C"
        return _run(in_5, in_0, in_1, in_2, in_4, in_3, in_6)