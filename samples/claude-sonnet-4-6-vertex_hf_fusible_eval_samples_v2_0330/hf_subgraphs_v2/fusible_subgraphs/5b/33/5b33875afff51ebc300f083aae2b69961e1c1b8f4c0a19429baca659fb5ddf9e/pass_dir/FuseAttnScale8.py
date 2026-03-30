"""
Fuse: matmul + permute(0,2,1,3) into a single Triton kernel that writes
directly in the permuted-contiguous [B,S,H,D] layout, eliminating the
permute + contiguous copy step.

The softmax/div/add chain remains in the graph.
The trailing contiguous() becomes a cheap no-op on an already-contiguous tensor.

Pattern matched:
  out = torch.matmul(x, y)          # x=[B,H,S,S], y=[B,H,S,D]
  return out.permute(0, 2, 1, 3)    # → [B,S,H,D]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: batched matmul writing directly in [B,S,H,D] layout
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_permuted_kernel(
    a_ptr,       # [B, H, S, S]  left operand (attn weights, post-softmax)
    b_ptr,       # [B, H, S, D]  right operand (value matrix)
    out_ptr,     # [B, S, H, D]  output in permuted-contiguous layout
    B, H, S, D,
    # strides for a [B, H, S, S]
    a_sb, a_sh, a_sq, a_sk,
    # strides for b [B, H, S, D]
    b_sb, b_sh, b_ss, b_sd,
    # strides for out [B, S, H, D]
    o_sb, o_ss, o_sh, o_sd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program → one (b, h, q) row of the output."""
    pid = tl.program_id(0)
    b = pid // (H * S)
    tmp = pid % (H * S)
    h = tmp // S
    q = tmp % S

    # load attention weights: a[b, h, q, :]
    a_base = a_ptr + b * a_sb + h * a_sh + q * a_sq
    k_range = tl.arange(0, BLOCK_S)
    k_mask = k_range < S

    a_row = tl.load(a_base + k_range * a_sk, mask=k_mask, other=0.0).to(tl.float32)

    # load value matrix and compute weighted sum
    b_base = b_ptr + b * b_sb + h * b_sh
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    vals = tl.load(
        b_base + k_range[:, None] * b_ss + d_range[None, :] * b_sd,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(a_row[:, None] * vals, axis=0)  # [BLOCK_D]

    # store to out[b, q, h, :] — permuted layout
    out_base = out_ptr + b * o_sb + q * o_ss + h * o_sh
    tl.store(
        out_base + d_range * o_sd,
        acc.to(out_ptr.dtype.element_ty),
        mask=d_range < D,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def matmul_permuted(x, y):
    B, H, S, _ = x.shape
    D = y.shape[-1]
    BLOCK_S = triton.next_power_of_2(S)
    BLOCK_D = triton.next_power_of_2(D)
    out = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    _matmul_permuted_kernel[(B * H * S,)](
        x, y, out,
        B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(x, y):
    out = torch.matmul(x, y)
    return out.permute(0, 2, 1, 3)


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return matmul_permuted