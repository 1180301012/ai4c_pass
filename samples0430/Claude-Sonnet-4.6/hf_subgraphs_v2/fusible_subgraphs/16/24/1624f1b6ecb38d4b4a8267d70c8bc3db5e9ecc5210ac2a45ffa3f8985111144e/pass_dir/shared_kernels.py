"""
Shared Triton kernels and dispatch wrapper for ViViT embedding passes.
Imported by FuseEmbedLayerNorm.py and FuseLayerNorm.py via:
    from pass_dir.shared_kernels import shared_dispatch_kernel, replacement_func
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused patch-embed pre-processing
#   conv3d_out : [1, 768, 5, 14, 14]  contiguous
#   cls_token  : [1,  1,  768]         contiguous
#   pos_emb    : [1, 981, 768]         contiguous
#   out        : [1, 981, 768]
#
#   2-D tiled: conv3d loaded [BLOCK_H, BLOCK_R] (inner=BLOCK_R → coalesced)
#              then register-transposed [BLOCK_R, BLOCK_H] for the add.
#   HIDDEN=768 is divisible by BLOCK_H=32 → no col masking needed.
# ---------------------------------------------------------------------------
@triton.jit
def _preproc_2d_kernel(
    conv3d_ptr,
    cls_ptr,
    pos_emb_ptr,
    out_ptr,
    N_PATCHES: tl.constexpr,   # 980
    HIDDEN:    tl.constexpr,   # 768
    N_SEQ:     tl.constexpr,   # 981
    BLOCK_R:   tl.constexpr,   # seq  rows per block (32)
    BLOCK_H:   tl.constexpr,   # hidden cols per block (32)
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    row_mask = rows < N_SEQ    # Only need row mask (768 % 32 == 0)
    # col_mask not needed: HIDDEN=768 is exactly divisible by BLOCK_H=32

    patch = tl.maximum(rows - 1, 0)

    # conv3d: [BLOCK_H, BLOCK_R], inner=BLOCK_R → coalesced reads
    c_off  = cols[:, None] * N_PATCHES + patch[None, :]
    c_mask = (rows > 0)[None, :]               # only row mask needed
    cv     = tl.load(conv3d_ptr + c_off, mask=c_mask, other=0.0)
    cv_T   = tl.trans(cv)   # [BLOCK_R, BLOCK_H]

    cls_v  = tl.load(cls_ptr + cols)           # [BLOCK_H], no mask needed
    is_cls = (rows == 0)[:, None]
    vals   = tl.where(is_cls, cls_v[None, :], cv_T)  # [BLOCK_R, BLOCK_H]

    # pos_emb: [BLOCK_R, BLOCK_H], inner=BLOCK_H → coalesced reads
    p_off = rows[:, None] * HIDDEN + cols[None, :]
    pos_v = tl.load(pos_emb_ptr + p_off, mask=row_mask[:, None], other=0.0)
    vals  = vals + pos_v

    tl.store(out_ptr + p_off, vals, mask=row_mask[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: Layer-norm  (BLOCK_SIZE=1024 with masking since 768 < 1024)
# ---------------------------------------------------------------------------
@triton.jit
def _layernorm_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr,
    HIDDEN:     tl.constexpr,   # 768
    BLOCK_SIZE: tl.constexpr,   # 1024 (power-of-2 ≥ HIDDEN)
):
    row  = tl.program_id(0)
    j    = tl.arange(0, BLOCK_SIZE)
    mask = j < HIDDEN

    x     = tl.load(x_ptr + row * HIDDEN + j, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mf32  = mask.to(tl.float32)      # 1.0 for valid, 0.0 for padding

    mean  = tl.sum(x_f32, axis=0) / HIDDEN
    diff  = (x_f32 - mean) * mf32   # zero out padding lanes
    var   = tl.sum(diff * diff, axis=0) / HIDDEN
    rstd  = 1.0 / tl.sqrt(var + 1e-6)

    w = tl.load(w_ptr + j, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + j, mask=mask, other=0.0).to(tl.float32)

    out = diff * rstd * w + b
    tl.store(out_ptr + row * HIDDEN + j, out, mask=mask)


# ---------------------------------------------------------------------------
# Python-level wrappers
# ---------------------------------------------------------------------------
def _fused_preproc(conv3d_out, cls_token, pos_emb):
    N_SEQ     = 981
    HIDDEN    = 768
    N_PATCHES = 980
    BLOCK_R   = 32
    BLOCK_H   = 32   # HIDDEN (768) % BLOCK_H (32) == 0  ✓

    out = torch.empty((1, N_SEQ, HIDDEN),
                      dtype=conv3d_out.dtype, device=conv3d_out.device)

    grid = (triton.cdiv(N_SEQ, BLOCK_R), triton.cdiv(HIDDEN, BLOCK_H))  # (31, 24)

    _preproc_2d_kernel[grid](
        conv3d_out, cls_token, pos_emb, out,
        N_PATCHES=N_PATCHES, HIDDEN=HIDDEN, N_SEQ=N_SEQ,
        BLOCK_R=BLOCK_R, BLOCK_H=BLOCK_H,
        num_warps=8, num_stages=2,
    )
    return out


def _fused_layernorm(x, ln_weight, ln_bias):
    N_SEQ  = 981
    HIDDEN = 768

    out = torch.empty_like(x)

    _layernorm_kernel[(N_SEQ,)](
        x, out, ln_weight, ln_bias,
        HIDDEN=HIDDEN,
        BLOCK_SIZE=1024,
        num_warps=8, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def shared_dispatch_kernel(*args):
    route       = args[-1]
    actual_args = args[:-1]
    if route == "preproc":
        return _fused_preproc(*actual_args)
    elif route == "layernorm":
        return _fused_layernorm(*actual_args)


def replacement_func():
    return shared_dispatch_kernel