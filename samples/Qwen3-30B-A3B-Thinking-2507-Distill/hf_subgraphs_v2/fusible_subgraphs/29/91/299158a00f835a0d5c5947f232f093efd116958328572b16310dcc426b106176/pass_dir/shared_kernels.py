"""
Shared Triton kernel for mean(-2) reduction.
Dispatch overhead is the bottleneck for these tiny tensors.
We keep the wrapper as lean as possible:
  1. No routing string overhead — FuseMeanDimNeg2 calls triton_mean directly.
  2. Fixed grid (B,) — no dynamic dispatch.
  3. Only one kernel launch per forward pass.
"""

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# Mean-reduction kernel  (x.mean(-2)  →  [B, K])
# One program per batch element; each program reduces K=448 columns over T=49.
# BLOCK_K=512 ≥ K=448  (loads K columns, 64-wide T-block covers all 49 rows).
# num_warps=4  →  128 threads; 128×256 = 32768 elements per kernel launch.
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _mean_kernel(
    in_ptr, out_ptr,
    T:       tl.constexpr,    # 49
    K:       tl.constexpr,    # 448
    BLOCK_K: tl.constexpr,    # 512
    BLOCK_T: tl.constexpr,    # 64
):
    """
    out[b, k] = mean_t( in[b, t, k] )
    Grid: (B,) — one program per batch element.
    Loads [BLOCK_K, BLOCK_T]  →  reduces over T (axis=1)  →  [BLOCK_K].
    """
    pid_b = tl.program_id(0)

    k_offs = tl.arange(0, BLOCK_K)   # [BLOCK_K]
    k_mask = k_offs < K

    t_offs = tl.arange(0, BLOCK_T)   # [BLOCK_T], power-of-2
    t_mask = t_offs < T

    offsets = pid_b * T * K + t_offs[None, :] * K + k_offs[:, None]

    vals = tl.load(in_ptr + offsets,
                   mask=k_mask[:, None] & t_mask[None, :],
                   other=0.0).to(tl.float32)

    acc    = tl.sum(vals, axis=1)    # [BLOCK_K]
    mean_v = acc / T

    tl.store(out_ptr + pid_b * K + k_offs,
             mean_v.to(out_ptr.dtype.element_ty),
             mask=k_mask)


# ── Module-level cache: pre-compile once per dtype ───────────────────────────
_compiled = {}


@torch.fx.wrap
def triton_mean(x):
    """x:[B,T=49,K=448]  →  mean(-2)  →  [B,K=448]  (fast Triton path)"""
    B = x.shape[0]
    K = x.shape[2]   # always 448
    out = torch.empty((B, K), device=x.device, dtype=x.dtype)
    # T=49, BLOCK_K=512, BLOCK_T=64 are compile-time constants → no autotune needed
    _mean_kernel[(B,)](
        x, out,
        T=49, K=K, BLOCK_K=512, BLOCK_T=64,
        num_warps=4,
    )
    return out