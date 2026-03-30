import torch
import triton
import triton.language as tl

# ── Pattern ────────────────────────────────────────────────────────────────────
# in_0 = bias  [H]
# in_1 = weight [H]
# in_2, in_3 = hidden states [B, T, H]
#
# Using in_0.shape[0] lets FX constant-fold the reshape/normalized_shape to the
# concrete H value (768 or 16) when the pattern is traced against each graph.
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    # in_0 is the bias vector of shape [H]; shape[0] folds to a concrete Python
    # int (768 or 16) when FX traces the pattern with concrete-shaped tensors,
    # giving reshape(-1, 768) / (768,) that matches the literal in each model.
    h = in_0.shape[0]
    tmp_3 = tmp_2.reshape(-1, h)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (h,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel ──────────────────────────────────────────────────────────────
# One kernel handles any (HIDDEN_SIZE, BLOCK_SIZE) pair; Triton specialises at
# first call for each unique constexpr combination.
@triton.jit
def fused_add_layer_norm_kernel(
    in2_ptr,
    in3_ptr,
    weight_ptr,
    bias_ptr,
    out_reshaped_ptr,
    out_normed_ptr,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HIDDEN_SIZE
    row_start = row_idx * HIDDEN_SIZE

    # --- fused add ----------------------------------------------------------
    x2 = tl.load(in2_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x  = x2 + x3

    # Write tmp_3 (the reshaped / added tensor)
    tl.store(out_reshaped_ptr + row_start + col_offsets, x, mask=mask)

    # --- layer norm ---------------------------------------------------------
    # Compute mean (padded zeros don't affect the sum)
    x_fp32 = x.to(tl.float32)
    mean   = tl.sum(x_fp32, axis=0) / HIDDEN_SIZE

    # Variance – zero masked lanes so they don't bias the result
    diff   = tl.where(mask, x_fp32 - mean, 0.0)
    var    = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm  = diff * inv_std

    weight  = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias_v  = tl.load(bias_ptr  + col_offsets, mask=mask, other=0.0).to(tl.float32)
    out     = x_norm * weight + bias_v

    # Write tmp_4 (the layer-normed tensor)
    tl.store(out_normed_ptr + row_start + col_offsets, out.to(x.dtype), mask=mask)


# ── Buffer cache ───────────────────────────────────────────────────────────────
# Avoid repeated torch.empty allocations across forward passes.
_buf_cache: dict = {}


# ── Inner launcher (opaque to FX) ─────────────────────────────────────────────
# Returns a [2, N, H] tensor so that the outer (traced) function can produce
# two independent getitem nodes — one for tmp_3, one for tmp_4.
@torch.fx.wrap
def _run_fused_add_layer_norm(in_0, in_1, in_2, in_3):
    hidden_size = in_0.shape[0]
    N           = in_2.numel() // hidden_size

    # Cached allocation: allocate once, reuse every call.
    key = (N, hidden_size, in_2.dtype, in_2.device)
    buf = _buf_cache.get(key)
    if buf is None:
        buf = torch.empty((2, N, hidden_size), dtype=in_2.dtype, device=in_2.device)
        _buf_cache[key] = buf

    out_reshaped = buf[0]   # [N, H]
    out_normed   = buf[1]   # [N, H]

    # Smallest power-of-2 >= hidden_size
    block_size = 1
    while block_size < hidden_size:
        block_size <<= 1

    num_warps = max(1, block_size // 256)

    fused_add_layer_norm_kernel[(N,)](
        in_2,
        in_3,
        in_1,            # weight
        in_0,            # bias
        out_reshaped,
        out_normed,
        1e-5,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return buf


# ── Outer replacement (FX traces INTO this → 2 independent getitem nodes) ─────
def fused_add_layer_norm(in_0, in_1, in_2, in_3):
    buf = _run_fused_add_layer_norm(in_0, in_1, in_2, in_3)
    return buf[0], buf[1]   # FX sees: getitem(buf,0), getitem(buf,1)


def replacement_func():
    return fused_add_layer_norm