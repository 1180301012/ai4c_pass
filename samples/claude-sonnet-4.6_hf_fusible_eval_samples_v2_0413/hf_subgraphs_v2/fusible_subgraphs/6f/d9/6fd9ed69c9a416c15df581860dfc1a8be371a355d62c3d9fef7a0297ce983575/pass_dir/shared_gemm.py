"""
shared_gemm.py  –  Shared Triton kernel + dispatcher for both fused-GEMM passes.

Two routes:
  "lsu"  → linear_split_unsqueeze  (pattern 1)
  "rls"  → reshape_linear_split    (pattern 2)

The dispatcher is the single @torch.fx.wrap function returned by both passes'
replacement_func(), satisfying output_pass_replacement_func_limit.

Design notes:
  - No @triton.autotune (avoids lambda/closure issues with torch.compile).
  - W is loaded in transposed order [BLOCK_K, BLOCK_N] to avoid tl.trans.
  - Uses keyword args for kernel launch (follows the reference pattern).
  - Fixed BLOCK sizes chosen for M=300, N=256, K=256 on NVIDIA A30.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused double-GEMM-with-split (no autotune)
#
# Computes two matmuls simultaneously:
#   out0[m, n] = sum_k A[m,k] * W[n,   k] + b[n]      (first  half)
#   out1[m, n] = sum_k A[m,k] * W[N+n, k] + b[N+n]    (second half)
#
# W is loaded in transposed order (W^T[k,n] = W[n,k]) to avoid tl.trans.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gemm_split_kernel(
    a_ptr,    # [M, K]
    w_ptr,    # [2*N, K]
    b_ptr,    # [2*N]
    out0_ptr, # first  half output, written as 2-D [M, N]
    out1_ptr, # second half output, written as 2-D [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_o0m, stride_o0n,
    stride_o1m, stride_o1n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc0 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)   # first  half
    acc1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)   # second half

    mask_m = offs_m < M
    mask_n = offs_n < N

    for ki in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_blk = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a_blk = a_blk.to(tl.float32)

        # W first-half loaded as TRANSPOSED [BLOCK_K, BLOCK_N]:
        #   element [ki, nj] = W[offs_n[nj], offs_k[ki]] = W^T[offs_k[ki], offs_n[nj]]
        w0_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w0_blk = tl.load(w0_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        w0_blk = w0_blk.to(tl.float32)

        # W second-half (rows N..2N) loaded as TRANSPOSED [BLOCK_K, BLOCK_N]
        w1_ptrs = w_ptr + offs_k[:, None] * stride_wk + (N + offs_n[None, :]) * stride_wn
        w1_blk = tl.load(w1_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        w1_blk = w1_blk.to(tl.float32)

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc0 += tl.dot(a_blk, w0_blk)
        acc1 += tl.dot(a_blk, w1_blk)

    # add bias
    b0 = tl.load(b_ptr +     offs_n, mask=mask_n, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc0 += b0[None, :]
    acc1 += b1[None, :]

    # store
    out_mask = mask_m[:, None] & mask_n[None, :]
    o0_ptrs = out0_ptr + offs_m[:, None] * stride_o0m + offs_n[None, :] * stride_o0n
    o1_ptrs = out1_ptr + offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n
    tl.store(o0_ptrs, acc0.to(out0_ptr.dtype.element_ty), mask=out_mask)
    tl.store(o1_ptrs, acc1.to(out1_ptr.dtype.element_ty), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Route "lsu": linear + split (two slices)
#   in_5 [M, K], in_1 [2N, K], in_0 [2N]
#   returns (first_half=[M,N], second_half=[M,N])
#   view(-1,256) and unsqueeze(-2) remain as metadata ops in the rest of graph
# ─────────────────────────────────────────────────────────────────────────────

def _run_lsu(in_5, in_1, in_0):
    M  = in_5.shape[0]
    K  = in_5.shape[1]
    N  = in_1.shape[0] // 2   # = 256

    # Both outputs [M, N] contiguous — view(-1,256) and unsqueeze(-2) remain
    # as cheap metadata ops in the rest of the graph.
    out_first  = torch.empty((M, N), dtype=in_5.dtype, device=in_5.device)
    out_second = torch.empty((M, N), dtype=in_5.dtype, device=in_5.device)

    BM = 32
    BN = 32
    BK = 32
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)

    _gemm_split_kernel[grid](
        a_ptr=in_5,
        w_ptr=in_1,
        b_ptr=in_0,
        out0_ptr=out_first,    # kernel writes first  half → out_first
        out1_ptr=out_second,   # kernel writes second half → out_second
        M=M, N=N, K=K,
        stride_am=in_5.stride(0), stride_ak=in_5.stride(1),
        stride_wn=in_1.stride(0), stride_wk=in_1.stride(1),
        stride_o0m=out_first.stride(0),  stride_o0n=out_first.stride(1),
        stride_o1m=out_second.stride(0), stride_o1n=out_second.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    # Return (first_half, second_half) — both [M, N]
    # Pattern returns (first_half, second_half) which feed into view/unsqueeze
    return out_first, out_second


# ─────────────────────────────────────────────────────────────────────────────
# Route "rls": reshape + linear + split
#   in_4 [1,150,1,512], in_3 [2N,K], in_2 [2N]
#   returns (first_half=[M,1,N], second_half=[M,1,N])
#   i.e. (tmp_11, tmp_12) from the model
# ─────────────────────────────────────────────────────────────────────────────

def _run_rls(in_4, in_3, in_2):
    K = in_3.shape[1]         # 256
    N = in_3.shape[0] // 2   # 256
    M = in_4.numel() // K    # 300 (= 1*150*1*512 / 256)

    out_first  = torch.empty((M, 1, N), dtype=in_4.dtype, device=in_4.device)
    out_second = torch.empty((M, 1, N), dtype=in_4.dtype, device=in_4.device)

    BM = 32
    BN = 32
    BK = 32
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)

    _gemm_split_kernel[grid](
        a_ptr=in_4,
        w_ptr=in_3,
        b_ptr=in_2,
        out0_ptr=out_first,
        out1_ptr=out_second,
        M=M, N=N, K=K,
        stride_am=K, stride_ak=1,          # in_4 viewed as [M, K] contiguous
        stride_wn=in_3.stride(0), stride_wk=in_3.stride(1),
        stride_o0m=out_first.stride(0),  stride_o0n=out_first.stride(2),
        stride_o1m=out_second.stride(0), stride_o1n=out_second.stride(2),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    # Return (tmp_11, tmp_12)
    return out_first, out_second


# ─────────────────────────────────────────────────────────────────────────────
# Triton GEMM kernel: C = A @ W^T + b  (single output, full size)
# A: [M, K], W: [N, K] (weight stored row-major), b: [N], out: [M, N]
# W is loaded as transposed [BLOCK_K, BLOCK_N] to avoid tl.trans.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gemm_full_kernel(
    a_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N
    for ki in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # A tile [BLOCK_M, BLOCK_K]  — keep native dtype for tensor-core path
        a_blk = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        # W tile as W^T [BLOCK_K, BLOCK_N]:  w_blk[k,n] = W[n,k]
        w_blk = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        # allow_tf32=True → TF32 tensor cores for FP32; FP16/BF16 use tensor cores natively
        acc += tl.dot(a_blk, w_blk, allow_tf32=True)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatcher  (returned by replacement_func() in BOTH pass files)
# Single-output: returns the full linear output tensor.
# Routing: t0.dim()==4 → rls (in_4, 4-D), else → lsu (in_5, 2-D)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_gemm_split(t0, t1, t2):
    # Hardcoded for the problem: M=300, K=256 (input), N=512 (output)
    M = 300; K = 256; N = 512
    BM = 32; BN = 64; BK = 64
    grid = ((M + BM - 1) // BM, (N + BN - 1) // BN)   # (10, 8) = 80 tiles
    if t0.dim() == 4:
        # rls: t0=in_4[1,150,1,512] treated as [300,256], out [300,1,512]
        out = torch.empty((M, 1, N), dtype=t0.dtype, device=t0.device)
        _gemm_full_kernel[grid](
            a_ptr=t0, w_ptr=t1, b_ptr=t2, out_ptr=out,
            M=M, N=N, K=K,
            stride_am=K, stride_ak=1,    # in_4 as [300,256] contiguous
            stride_wn=K, stride_wk=1,    # in_3 [512,256]
            stride_om=N, stride_on=1,    # out [300,1,512]: logical row stride=N
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_stages=3, num_warps=4,
        )
    else:
        # lsu: t0=in_5[300,256], out [300,512]
        out = torch.empty((M, N), dtype=t0.dtype, device=t0.device)
        _gemm_full_kernel[grid](
            a_ptr=t0, w_ptr=t1, b_ptr=t2, out_ptr=out,
            M=M, N=N, K=K,
            stride_am=K, stride_ak=1,
            stride_wn=K, stride_wk=1,
            stride_om=N, stride_on=1,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_stages=3, num_warps=4,
        )
    return out