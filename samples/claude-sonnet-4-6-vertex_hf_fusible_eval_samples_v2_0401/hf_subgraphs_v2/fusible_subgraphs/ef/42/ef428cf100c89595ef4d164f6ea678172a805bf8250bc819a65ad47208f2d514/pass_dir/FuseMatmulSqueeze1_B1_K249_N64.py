"""
Fused Triton pass: torch.matmul(in_0, in_1).squeeze(1)

Input shapes:
  in_0 : [B, 1, K]   e.g. [1, 1, 249]  bfloat16 / float16
  in_1 : [B, K, N]   e.g. [1, 249, 64] bfloat16 / float16

The matmul produces [B, 1, N]; squeeze(1) yields [B, N].
We fuse both ops into one Triton kernel that writes [B, N] directly,
accumulating in float32 and auto-casting back to the input dtype on store.

Block strategy:
  BLOCK_N = 64   (covers all N=64 in one tile)
  BLOCK_K = 16   (small chunk → intermediate [16,64] float32 = 4 KB,
                  fits comfortably in registers with 2 warps)
  num_warps = 2  (64 threads = exactly matches BLOCK_N)
  Iterations    = ceil(249/16) = 16  (tight inner loop, no spill)
  Grid          = (B, 1)  = 1 CTA for B=1
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: must mirror model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel  (fixed sizes tuned for B=1, K=249, N=64)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_matmul_squeeze_kernel(
    a_ptr,          # [B, 1, K]  – treated as [B, K] since M=1
    b_ptr,          # [B, K, N]
    out_ptr,        # [B, N]
    B,
    K,
    N,
    BLOCK_N: tl.constexpr,   # = 64
    BLOCK_K: tl.constexpr,   # = 16
):
    pid_b = tl.program_id(0)   # batch index
    pid_n = tl.program_id(1)   # tile index along N

    # Column offsets for this CTA
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Float32 accumulator (one entry per output column)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Base pointers (contiguous layout assumed; ensured by .contiguous() in wrapper)
    a_base = pid_b * K           # in_0[b, 0, k] = a_ptr + b*K + k
    b_base = pid_b * (K * N)     # in_1[b, k, n] = b_ptr + b*K*N + k*N + n

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load BLOCK_K elements from the in_0 row vector: shape [BLOCK_K]
        a = tl.load(a_ptr + a_base + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        # Load a [BLOCK_K × BLOCK_N] tile from in_1 (rows are contiguous → coalesced)
        b_ptrs = b_ptr + b_base + k_offs[:, None] * N + n_offs[None, :]
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0).to(tl.float32)

        # acc[n] += Σ_k  a[k] * b[k, n]
        # Intermediate [BLOCK_K, BLOCK_N] = [16, 64] float32 = 4 KB → fits in registers
        acc += tl.sum(a[:, None] * b, axis=0)

    # Triton auto-casts float32 → bfloat16 / float16 based on out_ptr element type
    tl.store(out_ptr + pid_b * N + n_offs, acc, mask=n_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_matmul_squeeze(in_0, in_1):
    """
    Fused matmul([B,1,K] × [B,K,N]) + squeeze(dim=1) → [B, N].
    Accumulates in float32 then casts back to the input dtype.
    """
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    B = in_0.shape[0]
    K = in_0.shape[2]   # 249
    N = in_1.shape[2]   # 64

    BLOCK_N = 64
    BLOCK_K = 16

    out = torch.empty((B, N), dtype=in_0.dtype, device=in_0.device)

    grid = (B, triton.cdiv(N, BLOCK_N))

    _fused_matmul_squeeze_kernel[grid](
        in_0, in_1, out,
        B, K, N,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=2,        # 64 threads = 1 element per thread for BLOCK_N=64
        num_stages=1,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement function hook (zero-argument, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_matmul_squeeze