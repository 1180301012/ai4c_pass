import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: tile([1,1,1]) + cat + add + dropout(p=0) + layer_norm
# ---------------------------------------------------------------------------

def pattern(in_2, tmp_8, in_3, in_5, in_4):
    tmp_9  = in_2.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    return (tmp_12, tmp_13)


def replacement_args(in_2, tmp_8, in_3, in_5, in_4):
    return (in_2, tmp_8, in_3, in_5, in_4)


# ---------------------------------------------------------------------------
# Unified Triton kernel: 1-D (BLOCK_ROWS=1) and 2-D (BLOCK_ROWS>1) paths.
#
# 2-D path key insight:
#   The transposed conv output tmp_8 has strides [768*980, 1, 980].
#   Element [0, patch_i, channel_j] is at byte-offset patch_i + channel_j*980.
#
#   If we load in [BLOCK_D, BLOCK_ROWS] layout (outer=channel, inner=patch),
#   consecutive threads hold consecutive patches for the SAME channel →
#   their offsets differ by 1  →  COALESCED L2 reads.
#
#   After tl.trans() we work in [BLOCK_ROWS, BLOCK_D] (outer=patch, inner=channel),
#   so pos reads and out/ln_out stores are stride-1 per warp → COALESCED HBM r/w.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # 2-D primary (123 CTAs on A30, 8× coalesced conv reads, ~1 full wave)
        triton.Config({'BLOCK_ROWS': 8,  'BLOCK_D': 1024}, num_warps=8,  num_stages=1),
        # 2-D conservative (246 CTAs, 4× coalesced conv reads, ~2 full waves)
        triton.Config({'BLOCK_ROWS': 4,  'BLOCK_D': 1024}, num_warps=8,  num_stages=1),
        # 1-D fallback (981 CTAs, simple + coalesced pos/out, non-coalesced conv)
        triton.Config({'BLOCK_ROWS': 1,  'BLOCK_D': 1024}, num_warps=16, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def tile_cat_add_ln_kernel(
    cls_ptr,         # [B, 1, D]     contiguous cls_token (in_2)
    conv_ptr,        # [B, N_CONV, D] NON-contiguous transposed conv output (tmp_8)
    pos_ptr,         # [B, N_ROWS, D] contiguous position embeddings (in_3)
    w_ptr,           # [D]           LN weight
    b_ptr,           # [D]           LN bias
    out_ptr,         # [B, N_ROWS, D] output tmp_12
    ln_out_ptr,      # [B, N_ROWS, D] output tmp_13
    N_CLS,
    N_ROWS,
    D: tl.constexpr,
    conv_stride_i,   # stride of tmp_8 along the patch dimension  (= 1)
    conv_stride_j,   # stride of tmp_8 along the channel dimension (= N_CONV = 980)
    eps,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)

    if BLOCK_ROWS == 1:
        # ============================================================
        # 1-D path  (one row per CTA, original implementation)
        # ============================================================
        row  = block_id
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        is_cls   = row < N_CLS
        conv_row = tl.maximum(row - N_CLS, 0)

        cls_val  = tl.load(cls_ptr  + cols,
                           mask=mask & is_cls, other=0.0,
                           eviction_policy='evict_last')
        conv_val = tl.load(conv_ptr + conv_row * conv_stride_i + cols * conv_stride_j,
                           mask=mask & ~is_cls, other=0.0,
                           eviction_policy='evict_first')

        x_f32   = tl.where(is_cls, cls_val.to(tl.float32), conv_val.to(tl.float32))
        pos_f32 = tl.load(pos_ptr + row * D + cols,
                          mask=mask, other=0.0,
                          eviction_policy='evict_first').to(tl.float32)
        out_f32 = x_f32 + pos_f32
        tl.store(out_ptr + row * D + cols,
                 out_f32.to(cls_val.dtype), mask=mask,
                 eviction_policy='evict_first')

        mean   = tl.sum(out_f32, axis=0) / D
        diff   = out_f32 - mean
        diff_m = tl.where(mask, diff, 0.0)
        var    = tl.sum(diff_m * diff_m, axis=0) / D
        rstd   = tl.rsqrt(var + eps)
        norm   = diff * rstd

        w_v = tl.load(w_ptr + cols, mask=mask, other=1.0,
                      eviction_policy='evict_last').to(tl.float32)
        b_v = tl.load(b_ptr + cols, mask=mask, other=0.0,
                      eviction_policy='evict_last').to(tl.float32)
        tl.store(ln_out_ptr + row * D + cols,
                 (norm * w_v + b_v).to(cls_val.dtype), mask=mask,
                 eviction_policy='evict_first')

    else:
        # ============================================================
        # 2-D path  (BLOCK_ROWS rows per CTA, coalesced conv reads)
        # ============================================================
        row_base = block_id * BLOCK_ROWS

        # --- Load phase: [BLOCK_D, BLOCK_ROWS] layout ---
        # Inner dim = BLOCK_ROWS (patches) → consecutive threads access
        # consecutive patches for the same channel → COALESCED L2 reads.
        c = tl.arange(0, BLOCK_D)[:, None]     # [BLOCK_D,    1      ]
        r = tl.arange(0, BLOCK_ROWS)[None, :]  # [1,           BLOCK_ROWS]

        row_g    = row_base + r                 # [1, BLOCK_ROWS]  global row indices
        mask_d   = c < D                        # [BLOCK_D, 1]
        mask_r   = row_g < N_ROWS              # [1, BLOCK_ROWS]
        mask_2d  = mask_d & mask_r             # [BLOCK_D, BLOCK_ROWS]

        is_cls   = row_g < N_CLS               # [1, BLOCK_ROWS]
        conv_row = tl.maximum(row_g - N_CLS, 0)  # [1, BLOCK_ROWS]

        # Broadcast c to [BLOCK_D, BLOCK_ROWS]  (r*0 adds the BLOCK_ROWS dim)
        c_br = c + r * 0                        # [BLOCK_D, BLOCK_ROWS]

        # cls_token is contiguous [1, D]: element [c] at offset c
        cls_val  = tl.load(cls_ptr  + c_br,
                           mask=mask_2d & is_cls,  other=0.0)

        # conv: element [conv_row, c] at offset conv_row * stride_i + c * stride_j
        # For fixed c, consecutive r (= consecutive patches) → stride-1 → COALESCED.
        conv_off = conv_row * conv_stride_i + c * conv_stride_j  # [BLOCK_D, BLOCK_ROWS]
        conv_val = tl.load(conv_ptr + conv_off,
                           mask=mask_2d & ~is_cls, other=0.0)

        input_dr = tl.where(is_cls, cls_val, conv_val)  # [BLOCK_D, BLOCK_ROWS]

        # Free-in-register transpose → [BLOCK_ROWS, BLOCK_D]
        input_rd = tl.trans(input_dr)                    # [BLOCK_ROWS, BLOCK_D]

        # --- Compute / store phase: [BLOCK_ROWS, BLOCK_D] layout ---
        # Inner dim = BLOCK_D (channels) → consecutive threads access
        # consecutive channels for the same patch → COALESCED HBM r/w.
        r_t      = tl.arange(0, BLOCK_ROWS)[:, None]    # [BLOCK_ROWS, 1]
        c_t      = tl.arange(0, BLOCK_D)[None, :]       # [1,          BLOCK_D]

        row_g_t  = row_base + r_t                        # [BLOCK_ROWS, 1]
        mask_d_t = c_t < D                               # [1, BLOCK_D]
        mask_r_t = row_g_t < N_ROWS                     # [BLOCK_ROWS, 1]
        mask_out = mask_d_t & mask_r_t                   # [BLOCK_ROWS, BLOCK_D]

        # Coalesced position-embedding read
        pos_off = row_g_t * D + c_t                      # [BLOCK_ROWS, BLOCK_D]
        pos = tl.load(pos_ptr + pos_off, mask=mask_out, other=0.0,
                      eviction_policy='evict_first')

        x_f32   = input_rd.to(tl.float32)
        out_f32 = x_f32 + pos.to(tl.float32)             # [BLOCK_ROWS, BLOCK_D]

        # Coalesced out store (tmp_12)
        tl.store(out_ptr + pos_off,
                 out_f32.to(input_rd.dtype), mask=mask_out,
                 eviction_policy='evict_first')

        # Layer norm: reduce over axis=1 (channels) for each of BLOCK_ROWS rows
        mean   = (tl.sum(out_f32, axis=1) / D)[:, None]              # [BLOCK_ROWS, 1]
        diff   = out_f32 - mean                                        # [BLOCK_ROWS, BLOCK_D]
        diff_m = tl.where(mask_d_t, diff, 0.0)
        var    = (tl.sum(diff_m * diff_m, axis=1) / D)[:, None]      # [BLOCK_ROWS, 1]
        rstd   = tl.rsqrt(var + eps)
        norm   = diff * rstd                                           # [BLOCK_ROWS, BLOCK_D]

        # LN params: small tensors, will stay in L1 across all blocks
        w_v = tl.load(w_ptr + c_t, mask=mask_d_t, other=1.0,
                      eviction_policy='evict_last').to(tl.float32)    # [1, BLOCK_D]
        b_v = tl.load(b_ptr + c_t, mask=mask_d_t, other=0.0,
                      eviction_policy='evict_last').to(tl.float32)    # [1, BLOCK_D]

        # Coalesced ln_out store (tmp_13)
        tl.store(ln_out_ptr + pos_off,
                 (norm * w_v + b_v).to(input_rd.dtype), mask=mask_out,
                 eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Wrapper + outer traceable function
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _run_tile_cat_kernel(in_2, tmp_8, in_3, ln_weight, ln_bias):
    B, N_CLS, D = in_2.shape
    N_CONV      = tmp_8.shape[1]
    N_ROWS      = N_CLS + N_CONV

    stride_i = tmp_8.stride(1)   # = 1   (patch dim stride after transpose)
    stride_j = tmp_8.stride(2)   # = 980 (channel dim stride after transpose)

    out    = torch.empty((B, N_ROWS, D), dtype=in_2.dtype, device=in_2.device)
    ln_out = torch.empty((B, N_ROWS, D), dtype=in_2.dtype, device=in_2.device)

    # Grid depends on BLOCK_ROWS (chosen by autotune at runtime)
    grid = lambda meta: (triton.cdiv(N_ROWS, meta['BLOCK_ROWS']),)

    tile_cat_add_ln_kernel[grid](
        in_2, tmp_8, in_3,
        ln_weight, ln_bias,
        out, ln_out,
        N_CLS=N_CLS,
        N_ROWS=N_ROWS,
        D=D,
        conv_stride_i=stride_i,
        conv_stride_j=stride_j,
        eps=1e-6,
    )
    return out, ln_out


def fused_tile_cat_add_layernorm(in_2, tmp_8, in_3, in_5, in_4):
    result = _run_tile_cat_kernel(in_2, tmp_8, in_3, in_5, in_4)
    out    = result[0]
    ln_out = result[1]
    return out, ln_out


def replacement_func():
    return fused_tile_cat_add_layernorm