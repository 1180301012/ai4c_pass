"""
FusePermReshapeCatCast: Fuses permute+reshape+permute+reshape+cat+to(float16)

Model subgraph matched:
    x1 [1, K, N1=9]  --permute(2,0,1)--reshape(-1,3,384,384)--> tmp_2 [9,  3,384,384]
    x2 [1, K, N2=25] --permute(2,0,1)--reshape(-1,3,384,384)--> tmp_5 [25, 3,384,384]
    cat([tmp_5, tmp_2, x0], dim=0)  --> [35, 3, 384, 384]
    .to(dtype=float16)              --> [35, 3, 384, 384] fp16

Optimization:
  Reads directly from unfold outputs [1,K,N] using 2D block loads + tl.trans(),
  casts to fp16, and writes to final output in one pass.
  Eliminates 5 intermediate tensor allocations (two reshape outputs + cat buffer
  + to-cast buffer + associated memory copies).

  Read pattern: load [BLOCK_K, BLOCK_N] tile from src (contiguous loads per warp).
  Write pattern: store tl.trans() result [BLOCK_N, BLOCK_K] (contiguous per row).
  Both reads and writes are fully coalesced.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Transpose + Cast kernel
# Reads src[0, k, n]  (shape [1, K, N]) and writes
#       out[out_offset + n, k]  (shape [TOTAL, K] fp16)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 256}),
        triton.Config({'BLOCK_K': 128}),
        triton.Config({'BLOCK_K': 64}),
    ],
    key=['K'],
)
@triton.jit
def transpose_cast_kernel(
    src_ptr,
    out_ptr,
    K,           # C*H*W  (runtime)
    out_offset,  # starting batch index in output (runtime)
    N: tl.constexpr,       # number of patches (compile-time: 9 or 25)
    BLOCK_K: tl.constexpr, # tile in K (autotuned)
    BLOCK_N: tl.constexpr, # ≥ N, power-of-2 (compile-time: 16 or 32)
):
    k_pid   = tl.program_id(0)
    k_start = k_pid * BLOCK_K

    k_range = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    n_range = tl.arange(0, BLOCK_N)             # [BLOCK_N]

    k_mask = k_range < K
    n_mask = n_range < N

    # Load [BLOCK_K, BLOCK_N] tile.
    # src[0, k, n] → flat index k * N + n.
    # Consecutive n within a row → contiguous memory → coalesced reads.
    k_2d   = k_range[:, None]   # [BLOCK_K, 1]
    n_2d   = n_range[None, :]   # [1, BLOCK_N]
    src_off = k_2d * N + n_2d   # [BLOCK_K, BLOCK_N]
    src_mask = k_mask[:, None] & n_mask[None, :]

    vals = tl.load(src_ptr + src_off, mask=src_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

    # Cast and transpose → [BLOCK_N, BLOCK_K]
    vals_fp16 = vals.to(tl.float16)
    vals_T    = tl.trans(vals_fp16)  # [BLOCK_N, BLOCK_K]

    # Write out[out_offset + n, k] → flat (out_offset + n)*K + k.
    # Consecutive k in row → contiguous writes → coalesced.
    out_n = (out_offset + n_range)[:, None]  # [BLOCK_N, 1]
    out_k = k_range[None, :]                 # [1, BLOCK_K]
    out_off  = out_n * K + out_k             # [BLOCK_N, BLOCK_K]
    out_mask = n_mask[:, None] & k_mask[None, :]

    tl.store(out_ptr + out_off, vals_T, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Simple copy + cast kernel (used for x0: one patch, K elements)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def copy_cast_kernel(
    src_ptr,
    out_ptr,
    K,
    out_offset,       # runtime batch index
    BLOCK_K: tl.constexpr,
):
    k_pid   = tl.program_id(0)
    k_start = k_pid * BLOCK_K
    k       = k_start + tl.arange(0, BLOCK_K)
    mask    = k < K

    vals     = tl.load(src_ptr + k, mask=mask, other=0.0)
    vals_fp16 = vals.to(tl.float16)

    out_off = out_offset * K + k
    tl.store(out_ptr + out_off, vals_fp16, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (replacement function)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def perm_reshape_cat_cast(x1, x2, x0):
    """
    Fused replacement for:
        x1.permute(2,0,1).reshape(-1,3,384,384)       → patches from unfold(in_1)
        x2.permute(2,0,1).reshape(-1,3,384,384)       → patches from unfold(in_2)
        torch.cat([tmp5, tmp2, x0], dim=0).to(fp16)

    x1 : [1, K, N1=9 ] unfold output for in_1
    x2 : [1, K, N2=25] unfold output for in_2
    x0 : [1, 3, H, W ] original in_0
    """
    N1      = 9
    N2      = 25
    N_TOTAL = N1 + N2 + 1   # 35
    K       = 3 * 384 * 384  # 442368  (C*H*W)

    # Flat output buffer [N_TOTAL * K] fp16 — viewed as [N_TOTAL, K]
    out_flat = torch.empty((N_TOTAL * K,), dtype=torch.float16, device=x0.device)

    COPY_BK = 2048

    # --- Kernel 1: x2 [1,K,25] → out[0..24, :] ---
    grid_t = lambda meta: (triton.cdiv(K, meta['BLOCK_K']),)
    transpose_cast_kernel[grid_t](
        x2.contiguous(), out_flat,
        K, 0,        # out_offset=0 (rows 0..24)
        N=N2, BLOCK_N=32,
    )

    # --- Kernel 2: x1 [1,K,9] → out[25..33, :] ---
    transpose_cast_kernel[grid_t](
        x1.contiguous(), out_flat,
        K, N2,       # out_offset=25 (rows 25..33)
        N=N1, BLOCK_N=16,
    )

    # --- Kernel 3: x0 [K] → out[34, :] ---
    x0_flat = x0.reshape(-1)   # [K]
    grid_c  = (triton.cdiv(K, COPY_BK),)
    copy_cast_kernel[grid_c](
        x0_flat, out_flat,
        K, N2 + N1,  # out_offset=34
        BLOCK_K=COPY_BK,
    )

    return out_flat.view(N_TOTAL, 3, 384, 384)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern / replacement interface
# ─────────────────────────────────────────────────────────────────────────────

def pattern(x1, x2, x0):
    """
    Matches:
      x2.permute(2,0,1).reshape(-1,3,384,384) → tmp_5   (N2=25 patches from in_2)
      x1.permute(2,0,1).reshape(-1,3,384,384) → tmp_2   (N1=9  patches from in_1)
      torch.cat([tmp_5, tmp_2, x0], dim=0).to(dtype=float16)

    In the model:
      x1 ↔ tmp_0 = unfold(in_1, stride=192)  [1, K, 9]
      x2 ↔ tmp_3 = unfold(in_2, stride=288)  [1, K, 25]
      x0 ↔ in_0                               [1, 3, 384, 384]
    """
    tmp_1 = x1.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_4 = x2.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, x0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7


def replacement_args(x1, x2, x0):
    return (x1, x2, x0)


def replacement_func():
    return perm_reshape_cat_cast