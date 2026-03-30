import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: torch.matmul([1,1,K], [1,K,N])  →  squeeze(1)  →  [1,N]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: single CTA computes all N=64 outputs in one launch.
# BLOCK_N=64 == N → no n-mask.  BLOCK_K=256 > K=249 → one tile, k-masked.
# fp32 accumulation.  num_warps=8 → 256 threads → 256 concurrent memory ops
# to maximise DRAM latency hiding for this 32 KB problem.
# ---------------------------------------------------------------------------

@triton.jit
def gemv_fused_kernel(
    a_ptr,            # [K]
    b_ptr,            # [K, N]  row-major
    out_ptr,          # [N]
    K: tl.constexpr,             # 249  – compile-time constant enables full unroll
    N: tl.constexpr,             # 64   – compile-time constant (stride of B, N=2^6)
    BLOCK_N: tl.constexpr,       # 64
    BLOCK_K: tl.constexpr,       # tile size for inner loop
):
    n_offs = tl.arange(0, BLOCK_N)          # [0..63]  no mask needed (BLOCK_N==N)
    acc    = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Tile K into BLOCK_K-sized chunks; with K constexpr the loop trip-count
    # is known at compile time, enabling unrolling / SW-pipelining.
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Broadcast A[k_offs] across all N threads
        a_vals = tl.load(a_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        # Coalesced row-wise loads of B[k_offs, n_offs]
        b_ptrs = b_ptr + k_offs[:, None] * N + n_offs[None, :]
        b_vals = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)

    tl.store(out_ptr + n_offs, acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def matmul_squeeze_triton(in_0, in_1):
    """
    Fused matmul + squeeze (single Triton kernel).

    in_0  : [1, 1, K]  (bfloat16 or float16)
    in_1  : [1, K, N]  (bfloat16 or float16)
    return: [1, N]
    """
    K = in_0.shape[-1]   # 249
    N = in_1.shape[-1]   # 64

    a_flat = in_0.view(K)      # [K]    – zero-copy view
    b_flat = in_1.view(K, N)   # [K, N] – zero-copy view

    out      = torch.empty((1, N), dtype=in_0.dtype, device=in_0.device)
    out_flat = out.view(N)

    gemv_fused_kernel[(1,)](
        a_flat, b_flat, out_flat,
        K=K, N=N,
        BLOCK_N=64,
        BLOCK_K=32,      # 8 loop iterations → best SW-pipeline depth for K=249
        num_warps=4,     # 128 threads → optimal balance
        num_stages=2,    # double-buffer: overlap load(k+1) with compute(k)
    )

    return out


def replacement_func():
    return matmul_squeeze_triton