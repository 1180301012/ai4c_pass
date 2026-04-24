import torch
import triton
import triton.language as tl
import math


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matmul(in_1, in_0) → view(B, H*M, 20, 20)
#   in_0 : [B, H, K, 400]   (K = 400, N = 400)
#   in_1 : [B, H, M, K]
#   out  : [B, H*M, 20, 20]
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    # Matches:  matmul = in_1 @ in_0   (attention-style batched matmul)
    matmul = in_1 @ in_0
    return matmul


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_20x20")


# ── Triton batched-GEMM kernel (general 2-D tiles) ──────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'GROUP_M': 8}, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _batched_gemm_20x20(
    a_ptr, b_ptr, c_ptr,
    # 4-D strides for a (= in_1): [B, H, M, K]
    a_sb, a_sh, a_sm, a_sk,
    # 4-D strides for b (= in_0): [B, H, K, N]
    b_sb, b_sh, b_sk, b_sn,
    # 4-D strides for c (= out):  [B, H, M, N]
    c_sb, c_sh, c_sm, c_sn,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    bh = tl.program_id(0)   # flattened batch index  (0 … B*H-1)
    rm = tl.program_id(1)   # tile row  index
    rn = tl.program_id(2)   # tile col  index

    # base row/col offsets
    rm_off = rm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_off = rn * BLOCK_N + tl.arange(0, BLOCK_N)

    a_base = a_ptr + bh * a_sb
    b_base = b_ptr + bh * b_sb
    c_base = c_ptr + bh * c_sb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_base + rm_off[:, None] * a_sm + rk[None, :] * a_sk,
            mask=(rm_off[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + rk[:, None] * b_sk + rn_off[None, :] * b_sn,
            mask=(rk[:, None] < K) & (rn_off[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        c_base + rm_off[:, None] * c_sm + rn_off[None, :] * c_sn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rm_off[:, None] < M) & (rn_off[None, :] < N),
    )


@torch.fx.wrap
def triton_attn_matmul_view_20x20(in_0, in_1):
    """
    Fused kernel: dispatch to shared Triton GEMM kernel (route_20x20).
    Allocates output in final [B, H*M, N, N] shape directly (view is free).
    """
    from pass_dir.dispatch_kernels import dispatch_kernel
    return dispatch_kernel(in_0, in_1, "route_20x20")


def replacement_func():
    from pass_dir.dispatch_kernels import dispatch_kernel
    return dispatch_kernel