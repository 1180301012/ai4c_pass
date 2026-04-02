import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: cat + add + dropout(p=0) + layer_norm
# Inputs   : tmp_9  = cls_token  [B, 1, D]   (result of tile, contiguous)
#            tmp_8  = conv embed [B, N, D]    (transposed, NON-contiguous)
#            in_3   = pos_embed  [B, N+1, D]  (contiguous)
#            in_5   = LN weight  [D]
#            in_4   = LN bias    [D]
# Outputs  : (tmp_12, tmp_13)  — both [B, N+1, D]
# ---------------------------------------------------------------------------

def pattern(tmp_9, tmp_8, in_3, in_5, in_4):
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    return (tmp_12, tmp_13)


def replacement_args(tmp_9, tmp_8, in_3, in_5, in_4):
    return (tmp_9, tmp_8, in_3, in_5, in_4)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_D': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_D': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_D': 1024}, num_warps=32, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def fused_cat_add_ln_kernel(
    cls_ptr,         # base ptr for tmp_9  [B, 1, D]  — contiguous, stride-1 in col
    conv_ptr,        # base ptr for tmp_8  [B, N, D]  — NON-contiguous (stride-based)
    pos_ptr,         # base ptr for in_3   [B, N+1, D] — contiguous
    w_ptr,           # LN weight [D]
    b_ptr,           # LN bias   [D]
    out_ptr,         # output tmp_12 [B, N+1, D]  — contiguous
    ln_out_ptr,      # output tmp_13 [B, N+1, D]  — contiguous
    N_CLS,           # = 1
    N_ROWS,          # = N_CLS + N_CONV  (= 981)
    D: tl.constexpr, # = 768
    conv_stride_i,   # tmp_8 stride along row dim   (= 1 for transposed tensor)
    conv_stride_j,   # tmp_8 stride along col dim   (= 980 for transposed tensor)
    eps,
    BLOCK_D: tl.constexpr,  # >= D, power of 2  (= 1024)
):
    """
    One CTA per output row.  For row 0 reads from cls_token; for rows 1..N-1
    reads from the non-contiguous conv output using gather-style strides.
    Fuses: virtual cat  →  add  →  identity-dropout  →  layer_norm.
    Both the add result (tmp_12) and the LN result (tmp_13) are stored.
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    is_cls   = row < N_CLS                     # True only for row 0
    conv_row = tl.maximum(row - N_CLS, 0)      # safe 0-based index into conv rows

    # ---- Load input element (cls or conv) ----
    # cls_ptr: element [0, 0, j] is at offset j (contiguous [1,1,D])
    cls_val  = tl.load(cls_ptr  + cols,
                       mask=mask & is_cls, other=0.0)
    # conv_ptr: element [0, i, j] is at offset i*stride_i + j*stride_j
    #   (for transposed [1,980,768]:  stride_i=1, stride_j=980)
    conv_val = tl.load(conv_ptr + conv_row * conv_stride_i + cols * conv_stride_j,
                       mask=mask & ~is_cls, other=0.0)

    x_f32 = tl.where(is_cls,
                     cls_val.to(tl.float32),
                     conv_val.to(tl.float32))

    # ---- Position embedding (contiguous [1, N_ROWS, D]) ----
    pos_f32 = tl.load(pos_ptr + row * D + cols,
                      mask=mask, other=0.0,
                      eviction_policy='evict_first').to(tl.float32)

    # ---- Add  (= cat + add + identity dropout) ----
    out_f32 = x_f32 + pos_f32

    # Store tmp_12  (evict quickly; readers of tmp_12 come from Python side)
    tl.store(out_ptr + row * D + cols,
             out_f32.to(cls_val.dtype), mask=mask,
             eviction_policy='evict_first')

    # ---- Layer Norm ----
    # Mean: masked padding is 0.0, so does not bias the sum
    mean = tl.sum(out_f32, axis=0) / D

    # Variance: explicitly zero out padded positions
    diff       = out_f32 - mean
    diff_m     = tl.where(mask, diff, 0.0)
    var        = tl.sum(diff_m * diff_m, axis=0) / D
    rstd       = tl.rsqrt(var + eps)
    norm       = diff * rstd

    # LN affine parameters  (small: kept in L1 across all rows)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0,
                eviction_policy='evict_last').to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0,
                eviction_policy='evict_last').to(tl.float32)

    # Store tmp_13
    tl.store(ln_out_ptr + row * D + cols,
             (norm * w + b).to(cls_val.dtype), mask=mask,
             eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Wrapper  (must be traceable by FX so we can return 2 getitem nodes)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _run_fused_cat_kernel(tmp_9, tmp_8, in_3, ln_weight, ln_bias):
    """
    Opaque (non-traced) function that sets up and launches the Triton kernel.
    Returns (out, ln_out) — both [B, N_ROWS, D].
    """
    B, N_CLS, D = tmp_9.shape      # [1, 1, 768]
    N_CONV      = tmp_8.shape[1]   # 980
    N_ROWS      = N_CLS + N_CONV   # 981

    # Strides of the non-contiguous transposed tensor tmp_8
    stride_i = tmp_8.stride(1)     # row stride  = 1
    stride_j = tmp_8.stride(2)     # col stride  = 980

    out    = torch.empty((B, N_ROWS, D), dtype=tmp_9.dtype, device=tmp_9.device)
    ln_out = torch.empty((B, N_ROWS, D), dtype=tmp_9.dtype, device=tmp_9.device)

    fused_cat_add_ln_kernel[(N_ROWS,)](
        tmp_9, tmp_8, in_3,
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


def fused_cat_add_layernorm(tmp_9, tmp_8, in_3, in_5, in_4):
    """
    Traceable outer replacement.  FX sees:
        result  = _run_fused_cat_kernel(...)  # opaque call
        out     = result[0]                    # getitem → tmp_12
        ln_out  = result[1]                    # getitem → tmp_13
        return (out, ln_out)                   # 2 returning nodes ✓
    """
    result = _run_fused_cat_kernel(tmp_9, tmp_8, in_3, in_5, in_4)
    out    = result[0]
    ln_out = result[1]
    return out, ln_out


def replacement_func():
    return fused_cat_add_layernorm