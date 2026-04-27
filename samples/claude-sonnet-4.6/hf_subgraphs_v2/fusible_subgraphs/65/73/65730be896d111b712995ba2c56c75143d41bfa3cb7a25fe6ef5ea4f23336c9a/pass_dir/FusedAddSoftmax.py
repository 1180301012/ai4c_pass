import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: add  →  .float()  →  softmax(dim=-1)  →  .type_as()  →  dropout(training=False)
# Use functional add (z = in_1 + in_0) so the result is a named node that
# can be used by both .float() and .type_as() — matching the model graph where
# in_2 = in_1 (alias) references the add result for both calls.
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    z = in_1 + in_0
    tmp_1 = z.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(z)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fuse add + float32 softmax + cast-back, all in one pass
# Each row of the last dimension is assigned to one program.
# BLOCK_N must be >= n_cols (enforced by the wrapper via next_power_of_2).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_add_softmax_kernel(
    x_ptr,            # in_0
    y_ptr,            # in_1
    out_ptr,          # output  (same dtype as in_1)
    n_cols,           # size of the last dimension
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols
    base = row_id * n_cols

    # ── Load + add (accumulate in fp32 for numerical stability) ──────────────
    x = tl.load(x_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    # ── Numerically stable softmax ────────────────────────────────────────────
    # Mask out padding lanes with -inf so they don't corrupt max/sum
    neg_inf = float('-inf')
    z_max = tl.max(tl.where(mask, z, neg_inf), axis=0)

    z_shifted = z - z_max                       # shift before exp
    z_exp = tl.exp(z_shifted)
    z_exp = tl.where(mask, z_exp, 0.0)          # zero out padding
    z_sum = tl.sum(z_exp, axis=0)
    result = z_exp / z_sum                       # normalised probabilities (fp32)

    # ── Store – Triton casts fp32 → out_ptr dtype automatically ──────────────
    tl.store(out_ptr + base + col_offsets, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper (must be decorated with @torch.fx.wrap)
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_fused_add_softmax(in_0, in_1):
    n_cols = in_0.shape[-1]
    n_rows = in_0.numel() // n_cols

    # Output tensor: same shape & dtype as in_1 (honours the type_as cast)
    out = torch.empty_like(in_1)

    # Choose the smallest power-of-2 block that covers all columns
    BLOCK_N = triton.next_power_of_2(n_cols)

    # num_warps: 1 warp for small blocks, scale up for larger ones
    num_warps = max(1, min(4, BLOCK_N // 32))

    _fused_add_softmax_kernel[(n_rows,)](
        in_0, in_1, out,
        n_cols=n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# replacement_func: return the callable (NOT a call)
# ─────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return triton_fused_add_softmax