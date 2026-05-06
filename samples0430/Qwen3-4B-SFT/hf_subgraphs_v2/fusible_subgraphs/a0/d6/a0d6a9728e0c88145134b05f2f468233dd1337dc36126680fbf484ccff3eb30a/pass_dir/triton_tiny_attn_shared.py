"""
Shared Triton kernel for fused tiny attention:
  Q @ K_T    ->  softmax  ->  attn @ V
where shapes are [B, H, 1] @ [B, 1, D] -> softmax -> @ [B, 1, D]
Result stored directly as [1, 1, B*H] (view+transpose+reshape fused in).

Works for H=32 (B=8) and H=64 (B=16), and all dtypes.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_D': 32}),
        triton.Config({'BLOCK_H': 64, 'BLOCK_D': 64}),
    ],
    key=['H', 'D'],
)
@triton.jit
def _tiny_attn_fused_kernel(
    q_ptr,            # [B_samples, H, 1]  – contiguous, strides (H, 1, 1)
    k_ptr,            # [B_samples, H, 1]
    v_ptr,            # [B_samples, 1, D]  – contiguous, strides (D, D, 1)
    out_ptr,          # [B_samples, H]     – contiguous, strides (H, 1)
    B_samples,        # runtime: number of batch rows
    H,                # runtime: hidden dim
    D,                # runtime: value dim
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    offs_h = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    offs_d = tl.arange(0, BLOCK_D)  # [BLOCK_D]

    h_mask = offs_h < H

    # ---- Load Q: shape [H] (row) ----
    # q[batch_idx, h, 0] = q_ptr[batch_idx * H + h]
    q = tl.load(q_ptr + batch_idx * H + offs_h, mask=h_mask, other=0.0)

    # ---- Scores = Q @ K_T : dot product per head ----
    # k[batch_idx, h, 0] = k_ptr[batch_idx * H + h]
    k = tl.load(k_ptr + batch_idx * H + offs_h, mask=h_mask, other=0.0)
    # scores[h, d] = sum_k q[h,k]*k[k,d]  →  reduce over k dimension (size 1!)
    # since K=1, scores[h] = q[h,0]*k[0,h]  (scalar per head)
    # Use tl.dot([H,1] @ [1,H]) = [H,H] then sum only column 0.
    # Simpler: scores[h] = dot(q[h,:], k[:,h]) — both are [H] over dim-0
    # k_T[h,0] = k[0,h]; scores[h] = sum_k q[k]*k_T[h,k] = sum_k q[k]*k[k,h]
    q_ext = tl.expand_dims(q, 1)  # [H, 1]
    k_T   = tl.expand_dims(k, 0)  # [1, H]
    scores = tl.sum(q_ext * k_T, axis=1)   # [H]  (K=1, so no tl.dot needed)

    # ---- Softmax (numerically stable) ----
    max_s    = tl.max(scores, axis=0)       # scalar
    exp_s    = tl.exp(scores - max_s)       # [H]
    sum_exp  = tl.sum(exp_s, axis=0)        # scalar
    attnWT   = exp_s / sum_exp              # [H]

    # ---- Load V: shape [D]  (v[batch,0,d] = v_ptr[batch*D + d]) ----
    v = tl.load(v_ptr + batch_idx * D + offs_d)  # [BLOCK_D]

    # ---- Output[h, d] = attnWT[h] * V[d]:  [H, D] matmul with H=1 row ----
    # attnWT[:, None] @ v[None, :] = [H, D] element-wise scaled sum
    out = tl.sum(tl.expand_dims(attnWT, 0) * tl.expand_dims(v, 1), axis=1)  # [H]

    # ---- Store directly in the final [1, 1, B*H] layout (flat offset = batch*H + h) ----
    tl.store(out_ptr + batch_idx * H + offs_h, out, mask=h_mask)


@torch.fx.wrap
def fused_tiny_attn_b8_h32_d32(q, k, v):
    """Fused kernel wrapper for B=8, H=32, D=32."""
    out = torch.empty(1, 1, 256, dtype=q.dtype, device=q.device)
    _tiny_attn_fused_kernel[(8,)](q, k, v, out, 8, 32, 32)
    return (out,)


@torch.fx.wrap
def fused_tiny_attn_b16_h64_d64(q, k, v):
    """Fused kernel wrapper for B=16, H=64, D=64."""
    out = torch.empty(1, 1, 1024, dtype=q.dtype, device=q.device)
    _tiny_attn_fused_kernel[(16,)](q, k, v, out, 16, 64, 64)
    return (out,)