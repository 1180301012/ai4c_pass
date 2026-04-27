"""
Shared Triton kernel for Swin Transformer attention bias + softmax fusion.

Fuses: gather(linear_out, indices) → sigmoid × 16 → add(in_2) → add(2×in_3) → softmax
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['B', 'H', 'N'],
)
@triton.jit
def _fused_gather_bias_softmax(
    linear_out_ptr,  # [K, H], K=225, H=num_heads  (float16 or bfloat16)
    indices_ptr,     # [N*N] int64 indices
    in2_ptr,         # [B, H, N, M]  attention scores
    in3_ptr,         # [B, N, M]     attention mask
    out_ptr,         # [B, H, N, M]  output
    B, H, N,
    M: tl.constexpr,  # = 64
):
    # Each program handles one row of the softmax: (b, h, n) → N key positions
    pid = tl.program_id(0)
    b = pid // (H * N)
    h = (pid % (H * N)) // N
    n = pid % N

    cols = tl.arange(0, M)            # [0..M-1]
    flat_pos = n * M + cols            # flat index into the N×M grid

    # ── Step 1: Gather position bias from linear_out ──────────────────────────
    # in_0 values are gather indices into linear_out rows
    gather_idx = tl.load(indices_ptr + flat_pos)          # [M] int64
    # linear_out is [K, H] row-major, stride along first dim = H
    lin_offset = gather_idx * H + h                        # [M]
    pos_raw = tl.load(linear_out_ptr + lin_offset).to(tl.float32)   # [M] f32
    pos_bias = 16.0 * tl.sigmoid(pos_raw)                 # [M] f32

    # ── Step 2: Load attention scores in_2[b, h, n, :] ───────────────────────
    in2_base = (b * H + h) * (N * M) + n * M
    in2_raw = tl.load(in2_ptr + in2_base + cols)          # [M] bf16/f16
    in2_vals = in2_raw.to(tl.float32)                     # [M] f32

    # ── Step 3: Load attention mask in_3[b, n, :] ────────────────────────────
    in3_base = b * (N * M) + n * M
    in3_vals = tl.load(in3_ptr + in3_base + cols).to(tl.float32)    # [M] f32

    # ── Step 4: Fused add:  scores + pos_bias + 2 * mask ─────────────────────
    # (mask is added twice in the original graph → ×2)
    x = in2_vals + pos_bias + 2.0 * in3_vals             # [M] f32

    # ── Step 5: Numerically stable online softmax ─────────────────────────────
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum                             # [M] f32

    # ── Step 6: Store result (auto-converts to output dtype) ──────────────────
    tl.store(out_ptr + in2_base + cols, x_softmax.to(in2_raw.dtype))


@torch.fx.wrap
def fused_swin_attn(linear_out, in_0, in_2, in_3):
    """
    Fused Swin attention bias + softmax kernel.

    Args:
        linear_out: raw linear output [1, 15, 15, H]  (H = num_heads)
        in_0:       relative position index  [N, N]  int64
        in_2:       attention scores         [B, H, N, M]
        in_3:       attention mask           [B, N, M]

    Returns:
        tuple: (softmax output [B, H, N, M],)
    """
    B, H, N, M = in_2.shape          # e.g. (64,12,64,64) or (16,24,64,64)

    # Reshape linear_out → [K, H] (K = 15*15 = 225)
    linear_flat = linear_out.view(-1, H).contiguous()
    # Flatten index tensor → [N*N]
    indices_flat = in_0.view(-1).contiguous()
    # Ensure contiguous inputs
    in_2_c = in_2.contiguous()
    in_3_c = in_3.contiguous()

    out = torch.empty_like(in_2_c)

    grid = (B * H * N,)
    _fused_gather_bias_softmax[grid](
        linear_flat, indices_flat, in_2_c, in_3_c, out,
        B, H, N,
        M=M,
    )

    return (out,)