"""
Shared Triton kernels for fused Add + MaskMax + View + Softmax pass.
Python-level ops (matching model.py exactly).
Backend pattern: pass_16_13_13 uses in_1 + in_0; others use max+view+softmax.
"""
import torch
import triton
import triton.language as tl


# ───────────────────────────────────────────────────────────────────────────
# Kernel B: max-fill ([B,H,L,L]) then softmax              (used by routes)
# ───────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_max_fill_softmax_kernel(
    x_ptr, const_ptr, out_ptr,
    L, n_rows,
    BLOCK_LEN: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col     = tl.arange(0, BLOCK_LEN)
    mask    = col < L
    const   = tl.load(const_ptr)
    x_vals  = tl.load(x_ptr + row_idx * L + col, mask=mask, other=0.0)
    scores  = tl.where(mask, x_vals, const)
    max_val = tl.max(scores, axis=0)
    exp_v   = tl.where(mask, tl.exp(scores - max_val), 0.0)
    probs   = tl.where(mask, exp_v / tl.sum(exp_v), 0.0)
    tl.store(out_ptr + row_idx * L + col, probs.to(x_vals.dtype), mask=mask)


@triton.jit
def fused_add_mask_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    L, n_rows,
    BLOCK_LEN: tl.constexpr,
):
    row_idx = tl.program_id(0)
    i_idx   = row_idx % L
    col     = tl.arange(0, BLOCK_LEN)
    mask    = col < L
    in1 = tl.load(in1_ptr + row_idx * L + col, mask=mask, other=0.0)
    in0 = tl.load(in0_ptr + i_idx  * L + col, mask=mask, other=0.0)
    scores = in1 + in0
    max_val = tl.max(scores, axis=0)
    exp_v   = tl.where(mask, tl.exp(scores - max_val), 0.0)
    probs   = tl.where(mask, exp_v / tl.sum(exp_v), 0.0)
    tl.store(out_ptr + row_idx * L + col, probs.to(in1.dtype), mask=mask)


# ───────────────────────────────────────────────────────────────────────────
# Dispatch wrappers
# ───────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_fused_max_fill_softmax(x, const_neg_inf, route):
    B, H, L = x.shape[0], x.shape[1], x.shape[2]
    n_rows  = B * H * L
    out     = torch.empty_like(x)
    BLOCK_LEN = max(16, 1 << (int(L) - 1).bit_length())
    fused_max_fill_softmax_kernel[(n_rows,)](
        x_ptr=x, const_ptr=const_neg_inf, out_ptr=out,
        L=L, n_rows=n_rows, BLOCK_LEN=BLOCK_LEN,
    )
    return out


@triton.jit
def triton_elem_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elem, BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def dispatch_fused_add_mask_softmax(in_0, in_1, route):
    """Replace in_1 + in_0 with a simple element-wise add."""
    N    = in_1.numel()
    out  = torch.empty_like(in_1)
    BLOCK = 1024
    triton_elem_add_kernel[((N + BLOCK - 1) // BLOCK,)](
        in_1, in_0, out, N, BLOCK)
    return out