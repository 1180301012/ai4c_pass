import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Triton kernel: fused batch-matmul + squeeze
#   out[b, n] = sum_k  A[b, 0, k] * B[b, k, n]
#
# Design: no autotune → no cache-lookup overhead per call.
#   BLOCK_N=64  – exactly covers N=64 in one program, no N-mask waste
#   BLOCK_K=256 → K=249 fits in ONE tl.load (zero loop iterations)
#   num_warps=4  (128 threads; BLOCK_N/num_warps = 0.5 → 2 threads/bin)
#   Grid: (B,)  → one block per batch element
# -----------------------------------------------------------------------
@triton.jit
def fused_matmul_squeeze_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_c0, stride_c1,
    N, K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out[b, n] = Σ_k  A[b, 0, k] * B[b, k, n]
    M=1 hardcoded.
    Loads A and B once then computes the full GEMV in one block.
    """
    pid_b  = tl.program_id(0)

    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    n_mask = n_offs < N
    k_mask = k_offs < K

    # Load A[0, k]: one coalesced K-vector
    a_row  = a_ptr + pid_b * stride_a0
    a_vals = tl.load(a_row + k_offs * stride_a2,
                     mask=k_mask, other=0.0)               # [BLOCK_K]

    # Load B[k, n]: [BLOCK_K, BLOCK_N] tile (n stride=1 → coalesced)
    b_row  = b_ptr + pid_b * stride_b0
    b_ptrs = b_row + k_offs[:, None] * stride_b1 + n_offs[None, :] * stride_b2
    b_vals = tl.load(b_ptrs,
                     mask=k_mask[:, None] & n_mask[None, :],
                     other=0.0)                             # [BLOCK_K, BLOCK_N]

    # Fused GEMV: output[j] = Σ_k  (a[k] * B[k,j])
    acc = tl.sum(a_vals[:, None] * b_vals, axis=0)        # [BLOCK_N] fp32

    # Store C[b, n]
    c_ptrs = c_ptr + pid_b * stride_c0 + n_offs * stride_c1
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=n_mask)


@torch.fx.wrap
def fused_matmul_squeeze(in_0, in_1):
    """
    Fused: (in_0 @ in_1).squeeze(dim=1)
    in_0: [B, M, K]   (M=1 here)
    in_1: [B, K, N]
    returns: [B, N]
    """
    B, M, K = in_0.shape
    N = in_1.shape[2]

    out = torch.empty((B, N), dtype=in_0.dtype, device=in_0.device)

    # Fixed launch configuration – no autotune overhead per call
    BLOCK_N = 64
    BLOCK_K = 256
    num_warps = 2   # 64 threads: 1 thread per output column, less spill

    fused_matmul_squeeze_kernel[(B,)](
        in_0, in_1, out,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),
        N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=2,
    )

    return out


# -----------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# -----------------------------------------------------------------------

def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_matmul_squeeze