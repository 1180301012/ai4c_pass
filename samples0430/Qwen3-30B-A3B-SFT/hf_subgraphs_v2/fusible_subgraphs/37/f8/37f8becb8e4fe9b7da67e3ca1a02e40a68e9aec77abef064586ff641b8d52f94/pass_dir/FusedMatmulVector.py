import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches only the matmul (inputs already on CUDA in all variants)
# ---------------------------------------------------------------------------

def pattern(in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    return matmul


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ---------------------------------------------------------------------------
# Universal Triton kernel: out[m] = dot(A[m,:], B[:,0])
#   K and BLOCK_K are compile-time constants for full specialisation.
#   fp32 accumulation; result cast to output dtype via pointer type.
# ---------------------------------------------------------------------------

@triton.jit
def _dot_kernel(A_ptr, B_ptr, out_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    m   = tl.program_id(0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        # A[m, k..k+BK] — base pointer A_ptr, row m, column offset
        x = tl.load(A_ptr + m * K + offs, mask=mask, other=0.0).to(tl.float32)
        # B[k..k+BK, 0] — base pointer B_ptr, column 0 (B is [K,1])
        y = tl.load(B_ptr + offs,       mask=mask, other=0.0).to(tl.float32)
        acc += x * y
    tl.store(out_ptr + m, tl.sum(acc, 0).to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Wrapper – @torch.fx.wrap keeps it opaque to torch.fx tracing
# K is passed as constexpr so the binary is fully specialised per K value.
# ---------------------------------------------------------------------------

_BLOCK_K = 256    # 3 exact iters for K=768 (fp16/bf16); 5 exact iters for K=1152 (fp32) — zero mask
_WARPS  = 4


@torch.fx.wrap
def triton_matmul_vector(A, B):
    """
    Compute A @ B where A:[M,K] and B:[K,1], both on CUDA.
    Returns [M,1] with the same dtype as A.
    """
    M = A.shape[0]
    K = A.shape[1]
    out = torch.empty((M, 1), dtype=A.dtype, device='cuda')
    _dot_kernel[(M,)](A, B, out, K=K, BLOCK_K=_BLOCK_K, num_warps=_WARPS)
    return out


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_matmul_vector