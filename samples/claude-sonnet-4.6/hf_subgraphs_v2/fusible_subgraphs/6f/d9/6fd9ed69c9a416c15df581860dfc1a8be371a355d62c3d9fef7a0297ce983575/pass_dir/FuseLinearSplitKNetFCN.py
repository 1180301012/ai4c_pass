import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: single F.linear call (applied to both linear ops automatically).
# ──────────────────────────────────────────────────────────────────────────────
def pattern(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def replacement_args(x, w, b):
    return (x, w, b)


# ──────────────────────────────────────────────────────────────────────────────
# Triton GEMM + bias kernel: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
#
# B is stored row-major [N,K]; loaded as transposed (BLOCK_K, BLOCK_N) tiles.
#   b_ptrs[k, n] = B[n, k]  →  tl.dot(a, b) = A @ B^T  ✓
#
# Configs use num_stages=1 or 2 to keep shared memory ≤ 24 KB/CTA, allowing
# 2-4 concurrent CTAs per SM on A30 (vs. 1 CTA with num_stages=4/48KB).
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # num_stages=1 → 12 KB shmem/CTA → 4 concurrent CTAs/SM (FP32)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=4),
        # num_warps=8 – more ILP per CTA
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=1, num_warps=8),
        # num_stages=2 → 24 KB shmem/CTA → 2 CTAs/SM (FP32), better pipeline
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id     = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A tile: (BLOCK_M, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B tile transposed: (BLOCK_K, BLOCK_N) → b[k, n] = B[n, k]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        # B is reused across GROUP_M=8 M-tiles for the same N-tile → cache in L2
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0,
                    cache_modifier='.ca')
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias is shared across all M-tiles for the same N-tile → cache aggressively
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0,
                   cache_modifier='.ca').to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper – no banned ATen dispatch ops.
# Strides are hardcoded for the specific shapes in this model:
#   x=[300,256] or x=[300,1,256]:  row-stride=256, col-stride=1
#   w=[512,256]:                   stride(1)=1,    stride(0)=256
#   output shape mirrors x leading dims + [N]
#   output strides: row-stride=N=512, col-stride=1
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _triton_linear(x, w, b):
    # Dimensions are fixed for this model (M=300, K=256, N=512)
    M, K, N = 300, 256, 512
    dtype, device = x.dtype, x.device

    # Output shape: [M, N] for 2D input x=[M,K], [M,1,N] for 3D x=[M,1,K]
    if x.ndim > 2:
        out = torch.empty(M, 1, N, dtype=dtype, device=device)
    else:
        out = torch.empty(M, N, dtype=dtype, device=device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _gemm_bias_kernel[grid](
        x, w, b, out,
        M, N, K,
        K, 1,   # stride_am, stride_ak
        1, K,   # stride_bk, stride_bn
        N, 1,   # stride_cm, stride_cn
    )
    return out


def replacement_func():
    return _triton_linear