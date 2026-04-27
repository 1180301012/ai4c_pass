"""Shared kernels + universal dispatch for two fusion patterns."""
import torch
import triton
import triton.language as tl


# ── Kernel 1: fused (a + b) → softmax(dim=-1) ──────────────────────────────
@triton.jit
def _add_softmax_kernel(
    a_ptr, b_ptr, out_ptr,
    B, H, S,
    BLOCK_S: tl.constexpr,
):
    """Each program: one row scores[b,h,q,:] + mask[0,0,0,:] → softmax."""
    pid = tl.program_id(0)
    b = pid // (H * S)
    h = (pid % (H * S)) // S
    q = pid % S

    s_offs = tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    row_a = tl.load(
        a_ptr + b * H * S * S + h * S * S + q * S + s_offs,
        mask=s_mask, other=0.0
    ).to(tl.float32)
    row_b = tl.load(b_ptr + s_offs, mask=s_mask, other=0.0).to(tl.float32)
    row = row_a + row_b

    max_val = tl.max(row, axis=0)
    row = row - max_val
    exp_row = tl.exp(row)
    out_row = exp_row / tl.sum(exp_row, axis=0)

    tl.store(
        out_ptr + b * H * S * S + h * S * S + q * S + s_offs,
        out_row, mask=s_mask
    )


# ── Kernel 2: matmul(a,b) → permute(0,2,1,3) → contiguous  ─────────────────
@triton.jit
def _matmul_permute_kernel(
    a_ptr, b_ptr, out_ptr,
    B, H, Sq, Sk, D,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (b,h): load b_tile once, loop over all q rows."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    sk_offs = tl.arange(0, BLOCK_SK)
    sk_mask = sk_offs < Sk
    d_offs  = tl.arange(0, BLOCK_D)
    d_mask  = d_offs < D

    # Load value tile b[b,h,:Sk,:D] once  →  reused for all q
    b_base = b * H * Sk * D + h * Sk * D
    b_tile = tl.load(
        b_ptr + b_base + sk_offs[:, None] * D + d_offs[None, :],
        mask=sk_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    # Process each query row sequentially
    for q in range(Sq):
        a_row = tl.load(
            a_ptr + b * H * Sq * Sk + h * Sq * Sk + q * Sk + sk_offs,
            mask=sk_mask, other=0.0
        ).to(tl.float32)

        out_row = tl.sum(a_row[:, None] * b_tile, axis=0)  # [BLOCK_D]

        # Write to permuted [B, Sq, H, D] layout
        out_base = b * Sq * H * D + q * H * D + h * D
        tl.store(out_ptr + out_base + d_offs, out_row, mask=d_mask)


# ── Universal dispatch (SHARED replacement_func across both passes) ──────────
@torch.fx.wrap
def _universal(a, b):
    """
    Dispatch based on b's shape:
      b.shape[0]==1 and b.shape[1]==1  →  add+softmax  (b is mask [1,1,1,S])
      otherwise                         →  matmul+permute+contiguous
    """
    if b.shape[0] == 1 and b.shape[1] == 1:
        # ── add + softmax ────────────────────────────────────────────────────
        B, H, S, _ = a.shape
        out = torch.empty_like(a)
        BLOCK_S = triton.next_power_of_2(S)
        _add_softmax_kernel[(B * H * S,)](
            a, b, out,
            B, H, S,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return out
    else:
        # ── matmul + permute(0,2,1,3) + contiguous ───────────────────────────
        B, H, Sq, Sk = a.shape
        D = b.shape[-1]
        out = torch.empty((B, Sq, H, D), dtype=a.dtype, device=a.device)
        BLOCK_SK = triton.next_power_of_2(Sk)
        BLOCK_D  = triton.next_power_of_2(D)
        _matmul_permute_kernel[(B * H,)](
            a, b, out,
            B, H, Sq, Sk, D,
            BLOCK_SK=BLOCK_SK,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        return out