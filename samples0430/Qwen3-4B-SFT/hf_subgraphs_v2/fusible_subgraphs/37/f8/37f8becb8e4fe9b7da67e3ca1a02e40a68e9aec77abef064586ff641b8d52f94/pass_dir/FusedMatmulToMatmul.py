import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: match torch.matmul(in_2, in_3)  →  single output [B,1] fp* tensor
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_2, in_3):
    return torch.matmul(in_2, in_3)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: C[b] = Σ_k A[b,k] * B[k,0]
#   Per-row design, one program per output row.
#   BLOCK_K=256 divides evenly into K=768 (3 iters) and K=1152 (5 iters).
#   num_stages=2 pipelines loads and compute for better memory throughput.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _dot_kernel(
    A_ptr,                    # [M, K]
    B_ptr,                    # [K, 1] contiguous → B_ptr + k == w[k,0]
    C_ptr,                    # [M] fp32 output
    K,
    BLOCK_K: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)

    A = tl.load(A_ptr + pid * K + offs_k).to(tl.float32)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k_base in range(0, K, BLOCK_K):
        offs = k_base + offs_k
        mask = offs < K
        B    = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += A * B

    tl.store(C_ptr + pid, tl.sum(acc))


# ──────────────────────────────────────────────────────────────────────────────
# Kernel wrapper
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_matmul_2xK(A, B):
    """
    Replaces torch.matmul(A, B) for A:[B,K], B:[K,1].
    Returns [B,1] in A.dtype.
    """
    B_val = A.shape[0]
    K     = A.shape[1]

    C_fp32 = torch.empty((B_val,), dtype=torch.float32, device=A.device)
    _dot_kernel[(B_val,)](
        A, B, C_fp32, K,
        BLOCK_K=64, num_warps=4, num_stages=3,
    )
    return C_fp32.to(A.dtype).view(B_val, 1)


def replacement_func():
    return triton_matmul_2xK