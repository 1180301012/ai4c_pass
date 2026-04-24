"""
Shared Triton kernels for fused reshape + split + permute + transpose
of a linear projection output used in attention mechanisms.

The linear layer outputs [B, S, 1536] which is:
  - reshaped to [B, S, H, total_D] = [B, S, 8, 192]
  - split along dim=3 into [32, 32, 128] for Q, K, V heads
  - permuted to [B, H, S, D] for each head
  - K transposed to [B, H, D, S] (done outside via transpose)

Inputs:
  linear_out: [B, S, 1536] contiguous tensor  (CUDA, bf16/fp16/fp32)
  in_0:       [8, S, S]  (attention mask on CPU)
Outputs:
  q: [B, 8, S, 32]
  k: [B, 8, S, 32]
  v: [B, 8, S, 128]
  attn_mask: [8, S, S] on CUDA
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Q / K split-and-permute kernel
# Reads linear_out[b, s, h*192 + d]       (Q slice, d in [0, 32))
#         and linear_out[b, s, h*192 + 32 + d]  (K slice, d in [0, 32))
# Writes q[b, h, s, d] and k[b, h, s, d]
# All shape constants are tl.constexpr for compiler optimisation.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['B'],
)
@triton.jit
def _qk_split_permute_kernel(
    linear_ptr,
    q_ptr, k_ptr,
    B, S,
    D_q: tl.constexpr,    # 32
    D_k: tl.constexpr,    # 32
    D_v: tl.constexpr,    # 128
    H:   tl.constexpr,    # 8
    BLOCK_SIZE: tl.constexpr,
):
    # All shape constants derived from constexpr parameters (compile-time constants)
    D_qk    = D_q + D_k      # 64
    HxDqk   = H * D_qk       # 512
    HDqv    = H * D_v        # 1024
    SxDqv   = S * HDqv       # stride in B dim for linear_out
    Dtotal  = D_qk + D_v     # 192  (full output per seq position)

    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total   = B * S * HxDqk
    mask    = offsets < total

    # Decode flat index into (b, s, h, d) -- b most-significant
    rem   = offsets % (S * HxDqk)
    b_idx = (offsets - rem) // (S * HxDqk)
    s_idx = rem // HxDqk
    rem2  = rem % HxDqk
    h_idx = rem2 // D_qk
    d_idx = rem2 % D_qk

    # ---- Q ----
    q_mask = mask & (d_idx < D_q)
    q_off  = b_idx * (H * S * D_q) + h_idx * (S * D_q) + s_idx * D_q + d_idx
    # linear_out[b, s, h*192 + d]   (h*D_qk + d = h*64 + d for h in [0..7], d in [0..63])
    lin_q  = b_idx * (S * Dtotal) + s_idx * Dtotal + h_idx * D_qk + d_idx
    q_val  = tl.load(linear_ptr + lin_q, mask=q_mask, other=0.0)
    tl.store(q_ptr + q_off, q_val, mask=q_mask)

    # ---- K ----
    k_mask = mask & (d_idx >= D_q)
    k_off  = b_idx * (H * S * D_k) + h_idx * (S * D_k) + s_idx * D_k + (d_idx - D_q)
    lin_k  = b_idx * (S * Dtotal) + s_idx * Dtotal + h_idx * D_qk + D_q + (d_idx - D_q)
    k_val  = tl.load(linear_ptr + lin_k, mask=k_mask, other=0.0)
    tl.store(k_ptr + k_off, k_val, mask=k_mask)


# ---------------------------------------------------------------------------
# V permute kernel
# Reads linear_out[b, s, h*192 + 64 + d]  (V slice, d in [0, 128))
# Writes v[b, h, s, d]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['B'],
)
@triton.jit
def _v_permute_kernel(
    linear_ptr,
    v_ptr,
    B, S,
    D_v: tl.constexpr,    # 128
    H:   tl.constexpr,    # 8
    BLOCK_SIZE: tl.constexpr,
):
    HDv   = H * D_v          # 1024
    SxDv  = S * HDv

    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total   = B * S * HDv
    mask    = offsets < total

    rem   = offsets % (S * HDv)
    b_idx = (offsets - rem) // (S * HDv)
    rem2  = rem % HDv
    s_idx = rem2 // D_v
    h_idx = rem2 % HDv // D_v
    d_idx = rem2 % D_v

    # linear_out[b, s, h*192 + 64 + d]  = b*S*192 + s*192 + h*192 + 64 + d
    lin_off = b_idx * (S * (D_q + D_k + D_v)) + s_idx * (D_q + D_k + D_v) + h_idx * D_v + d_idx + D_q + D_k
    v_off   = b_idx * HDv * S + h_idx * S * D_v + s_idx * D_v + d_idx

    v_val = tl.load(linear_ptr + lin_off, mask=mask, other=0.0)
    tl.store(v_ptr + v_off, v_val, mask=mask)


# ---------------------------------------------------------------------------
# CPU-to-CUDA attention-mask copy kernel
# ---------------------------------------------------------------------------
@triton.jit
def _copy_cpu_to_cuda_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    val  = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, val, mask=mask)


# ---------------------------------------------------------------------------
# @torch.fx.wrap wrapper -- returned by replacement_func() in every pass file
# ---------------------------------------------------------------------------
@torch.fx.wrap
def qkv_fused_wrapper(linear_out, in_0):
    """
    Fused reshape + split + permute for Q/K/V + attn-mask CUDA transfer.

    Args:
        linear_out : [B, S, 1536]  (S=49), CUDA, bf16/fp16/fp32
        in_0       : [8, S, S]     CPU,  bf16/fp16/fp32  (attention mask)
    Returns:
        (q, attn_mask, k, v)  -- matches original graph's return order
        q   : [B, 8, S, 32]
        k   : [B, 8, S, 32]
        v   : [B, 8, S, 128]
    """
    B, S = linear_out.shape[0], 49
    H, D_q, D_k, D_v = 8, 32, 32, 128

    device  = linear_out.device
    dtype   = linear_out.dtype

    q  = torch.empty((B, H, S, D_q), dtype=dtype, device=device)
    k  = torch.empty((B, H, S, D_k), dtype=dtype, device=device)
    v  = torch.empty((B, H, S, D_v), dtype=dtype, device=device)
    # Move attention-mask from CPU to CUDA (same dtype, shape [8, 49, 49])
    attn_mask = torch.as_tensor(in_0, device=device, dtype=dtype)

    # Launch Q/K kernel
    n_qk = B * S * H * (D_q + D_k)
    _qk_split_permute_kernel[lambda META: (triton.cdiv(n_qk, META['BLOCK_SIZE']),)](
        linear_out, q, k, B, S, D_q, D_k, D_v, H,
    )

    # Launch V kernel
    n_v = B * S * H * D_v
    _v_permute_kernel[lambda META: (triton.cdiv(n_v, META['BLOCK_SIZE']),)](
        linear_out, v, B, S, D_v, H,
    )

    return q, attn_mask, k, v