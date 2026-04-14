"""Shared Triton kernels and dispatch wrapper for MPNet embedding+LayerNorm fusion."""
import torch
import triton
import triton.language as tl


# ── H=768 kernel — all constants hardcoded inside, zero constexpr args ────────
@triton.jit
def _emb_ln_h768(
    idx1_ptr, idx2_ptr,
    emb1_ptr, emb2_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
):
    BLOCK_H: tl.constexpr = 1024
    H:       tl.constexpr = 768
    eps:     tl.constexpr = 1e-05

    pid  = tl.program_id(0)
    idx1 = tl.load(idx1_ptr + pid)
    idx2 = tl.load(idx2_ptr + pid)

    h    = tl.arange(0, BLOCK_H)
    mask = h < H

    e1 = tl.load(emb1_ptr + idx1 * H + h, mask=mask, other=0.0)
    e2 = tl.load(emb2_ptr + idx2 * H + h, mask=mask, other=0.0)
    orig_dt = e1.dtype
    x = e1.to(tl.float32) + e2.to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / H
    xn   = diff * (1.0 / tl.sqrt(var + eps))

    w  = tl.load(weight_ptr + h, mask=mask, other=1.0).to(tl.float32)
    b_ = tl.load(bias_ptr   + h, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid * H + h, (xn * w + b_).to(orig_dt), mask=mask)


# ── H=64 kernel — all constants hardcoded inside ──────────────────────────────
@triton.jit
def _emb_ln_h64(
    idx1_ptr, idx2_ptr,
    emb1_ptr, emb2_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
):
    BLOCK_H: tl.constexpr = 64
    H:       tl.constexpr = 64
    eps:     tl.constexpr = 1e-12

    pid  = tl.program_id(0)
    idx1 = tl.load(idx1_ptr + pid)
    idx2 = tl.load(idx2_ptr + pid)

    h    = tl.arange(0, BLOCK_H)
    mask = h < H

    e1 = tl.load(emb1_ptr + idx1 * H + h, mask=mask, other=0.0)
    e2 = tl.load(emb2_ptr + idx2 * H + h, mask=mask, other=0.0)
    orig_dt = e1.dtype
    x = e1.to(tl.float32) + e2.to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / H
    xn   = diff * (1.0 / tl.sqrt(var + eps))

    w  = tl.load(weight_ptr + h, mask=mask, other=1.0).to(tl.float32)
    b_ = tl.load(bias_ptr   + h, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid * H + h, (xn * w + b_).to(orig_dt), mask=mask)


# ── Shared dispatch wrapper ───────────────────────────────────────────────────
# Cache output tensors and Triton launcher objects across calls.
_out_cache: dict = {}
_launcher_cache: dict = {}

@torch.fx.wrap
def dispatch_kernel(in_0, in_1, in_2, in_3, in_4, in_5, route):
    """
    in_0: [B,S]   word token IDs
    in_1: [H]     LayerNorm bias
    in_2: [H]     LayerNorm weight
    in_3: [V2,H]  position embedding table
    in_4: [V1,H]  word embedding table
    in_5: [B,S]   position IDs (always on CUDA)
    route: "h768_e5" | "h64_e12"
    """
    B  = in_0.shape[0]
    S  = in_0.shape[1]
    BS = B * S
    device = in_5.device
    dtype  = in_4.dtype

    if route == "h768_e5":
        okey = (B, S, 768, dtype, device)
        out  = _out_cache.get(okey)
        if out is None:
            out = torch.empty(B, S, 768, dtype=dtype, device=device)
            _out_cache[okey] = out

        lkey    = (BS, route)
        launcher = _launcher_cache.get(lkey)
        if launcher is None:
            launcher = _emb_ln_h768[(BS,)]
            _launcher_cache[lkey] = launcher
        launcher(in_0, in_5, in_4, in_3, in_2, in_1, out,
                 num_warps=8)
    else:                       # "h64_e12"
        okey = (B, S, 64, dtype, device)
        out  = _out_cache.get(okey)
        if out is None:
            out = torch.empty(B, S, 64, dtype=dtype, device=device)
            _out_cache[okey] = out

        lkey    = (BS, route)
        launcher = _launcher_cache.get(lkey)
        if launcher is None:
            launcher = _emb_ln_h64[(BS,)]
            _launcher_cache[lkey] = launcher
        launcher(in_0, in_5, in_4, in_3, in_2, in_1, out,
                 num_warps=2)
    return out