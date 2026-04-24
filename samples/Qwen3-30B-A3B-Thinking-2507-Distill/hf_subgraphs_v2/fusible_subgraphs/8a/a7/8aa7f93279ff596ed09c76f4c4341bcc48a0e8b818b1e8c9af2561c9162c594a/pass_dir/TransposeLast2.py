import torch
import triton
import triton.language as tl


# ─── Pattern: transpose last two dims ────────────────────────────────────────

def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x, "transpose_last2")


# ─── Triton kernel: tiled matrix transpose for [B, H, M, K] tensors ──────────
# Each program handles one (batch×head, M-tile) pair and transposes K columns.

@triton.jit
def _transpose_last2_kernel(
    in_ptr,
    out_ptr,
    BH, M, K,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    bh     = tl.program_id(0)
    m_pid  = tl.program_id(1)
    k_pid  = tl.program_id(2)

    m_start = m_pid * TILE_M
    k_start = k_pid * TILE_K

    m_offs = m_start + tl.arange(0, TILE_M)
    k_offs = k_start + tl.arange(0, TILE_K)
    m_mask = m_offs < M

    # Load [TILE_M, TILE_K] tile from input[bh, m, k]
    in_off  = bh * M * K + m_offs[:, None] * K + k_offs[None, :]
    in_mask = m_mask[:, None]
    block   = tl.load(in_ptr + in_off, mask=in_mask, other=0.0)

    # Store transposed [TILE_K, TILE_M] tile to output[bh, k, m]
    out_off  = bh * K * M + k_offs[:, None] * M + m_offs[None, :]
    out_mask = k_offs[None, :] < K
    tl.store(out_ptr + out_off, tl.trans(block), mask=out_mask)


def _run_transpose_last2(x):
    B  = x.shape[0]
    H  = x.shape[1]
    M  = x.shape[2]
    K  = x.shape[3]
    BH = B * H
    TILE_M = 64
    TILE_K = 32
    out = torch.empty(B, H, K, M, dtype=x.dtype, device=x.device)
    grid = (BH, (M + TILE_M - 1) // TILE_M, (K + TILE_K - 1) // TILE_K)
    _transpose_last2_kernel[grid](x, out, BH, M, K, TILE_M=TILE_M, TILE_K=TILE_K)
    return out


# ─── Placeholder for the scale route (never actually called here) ─────────────

def _noop_scale(x):
    raise RuntimeError("_noop_scale called - this is a compile-time stub")


# ─── Shared dispatch wrapper (IDENTICAL to the one in FuseScaleMul.py) ───────

@torch.fx.wrap
def _dispatch_wrapper(arg, route):
    if route == "scale_mul":
        return _noop_scale(arg)
    elif route == "transpose_last2":
        return _run_transpose_last2(arg)


def replacement_func():
    return _dispatch_wrapper