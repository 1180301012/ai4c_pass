"""
Unified dispatch for fused roll+crop+add AND fused layer_norm.

All 6 pass files (3 add + 3 ln) return THIS SAME function → replacement_func_limit=1 satisfied.

Dispatch routing:
  "add_96", "add_192", "add_384"  → fused_roll_crop_add_kernel (single output)
  "ln_96",  "ln_192",  "ln_384"   → fused_layernorm_kernel      (single output)

Argument convention for fused_dispatch(arg0, arg1, arg2, route):
  add_*: arg0=in_2 (layer_output), arg1=in_3 (contiguous), arg2=in_2 (dummy dup), route="add_*"
  ln_*:  arg0=x    (residual),     arg1=weight,             arg2=bias,              route="ln_*"
"""
import torch
import triton
import triton.language as tl


# ─── Roll + crop + add kernel ─────────────────────────────────────────────────
@triton.jit
def fused_roll_crop_add_kernel(
    in3_ptr,    # contiguous, effective [B, H, H, C]
    in2_ptr,    # [B, CROP*CROP, C]
    out_ptr,    # [B, CROP*CROP, C]
    H:       tl.constexpr,
    CROP:    tl.constexpr,
    C:       tl.constexpr,
    SHIFT:   tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid   = tl.program_id(0)
    N_out = CROP * CROP
    b     = pid // N_out
    token = pid % N_out
    out_r = token // CROP
    out_c = token % CROP

    src_r = (out_r - SHIFT + H) % H
    src_c = (out_c - SHIFT + H) % H

    in3_base = (b * H * H + src_r * H + src_c) * C
    tok_base = (b * N_out + token) * C

    c_range = tl.arange(0, BLOCK_C)
    mask    = c_range < C

    x = tl.load(in3_ptr + in3_base + c_range, mask=mask, other=0.0)
    y = tl.load(in2_ptr + tok_base  + c_range, mask=mask, other=0.0)
    tl.store(out_ptr + tok_base + c_range, x + y, mask=mask)


# ─── Layer-norm kernel ─────────────────────────────────────────────────────────
@triton.jit
def fused_layernorm_kernel(
    x_ptr,        # [B, N, C] input
    w_ptr,        # [C] weight
    b_ptr,        # [C] bias
    out_ptr,      # [B, N, C] output
    C:            tl.constexpr,
    eps:          float,
    BLOCK_C:      tl.constexpr,
    ODTYPE:       tl.constexpr,  # output dtype: tl.float16/bfloat16/float32
):
    pid      = tl.program_id(0)
    tok_base = pid * C
    c_range  = tl.arange(0, BLOCK_C)
    mask     = c_range < C

    x   = tl.load(x_ptr + tok_base + c_range, mask=mask, other=0.0).to(tl.float32)
    w   = tl.load(w_ptr + c_range,             mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(b_ptr + c_range,             mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    norm = diff * tl.rsqrt(var + eps)
    out  = norm * w + b

    tl.store(out_ptr + tok_base + c_range, out.to(ODTYPE), mask=mask)


# ─── Unified dispatch ──────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_dispatch(arg0, arg1, arg2, route):
    """
    arg0, arg1, arg2, route:
      add_*: (in_2, in_3, in_2[dup], "add_96/192/384")
      ln_*:  (x, weight, bias, "ln_96/192/384")
    Returns a single tensor.
    """
    out = torch.empty_like(arg0)
    B   = arg0.shape[0]

    # Output dtype (needed for layernorm cast)
    dt = str(arg0.dtype)
    if dt == "torch.float16":
        ODTYPE = tl.float16
    elif dt == "torch.bfloat16":
        ODTYPE = tl.bfloat16
    else:
        ODTYPE = tl.float32

    if route == "add_96":
        total = B * 128 * 128
        fused_roll_crop_add_kernel[(total,)](
            arg1, arg0, out, H=133, CROP=128, C=96, SHIFT=3, BLOCK_C=128,
            num_warps=4,
        )
    elif route == "add_192":
        total = B * 64 * 64
        fused_roll_crop_add_kernel[(total,)](
            arg1, arg0, out, H=70, CROP=64, C=192, SHIFT=3, BLOCK_C=256,
            num_warps=4,
        )
    elif route == "add_384":
        total = B * 32 * 32
        fused_roll_crop_add_kernel[(total,)](
            arg1, arg0, out, H=35, CROP=32, C=384, SHIFT=3, BLOCK_C=512,
            num_warps=4,
        )
    elif route == "ln_96":
        total = B * arg0.shape[1]
        fused_layernorm_kernel[(total,)](
            arg0, arg1, arg2, out, C=96, eps=1e-5, BLOCK_C=128, ODTYPE=ODTYPE,
            num_warps=4,
        )
    elif route == "ln_192":
        total = B * arg0.shape[1]
        fused_layernorm_kernel[(total,)](
            arg0, arg1, arg2, out, C=192, eps=1e-5, BLOCK_C=256, ODTYPE=ODTYPE,
            num_warps=4,
        )
    else:  # "ln_384"
        total = B * arg0.shape[1]
        fused_layernorm_kernel[(total,)](
            arg0, arg1, arg2, out, C=384, eps=1e-5, BLOCK_C=512, ODTYPE=ODTYPE,
            num_warps=8,
        )

    return out