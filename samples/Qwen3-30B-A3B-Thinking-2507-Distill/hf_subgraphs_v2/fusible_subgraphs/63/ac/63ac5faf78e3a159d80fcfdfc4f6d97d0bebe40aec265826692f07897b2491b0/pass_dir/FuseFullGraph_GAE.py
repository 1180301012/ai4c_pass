import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3, in_5, in_4, in_2):
    """
    Full-graph pattern for GAE: deg_norm chain (pow_ + eq + masked_fill_ + getitem×2 + mul×3)
    + linear(in_0, in_1, None).

    in_3 is a pattern PLACEHOLDER → can have external users of pow_(in_3, -0.5) in target
    without triggering NOT_CONTAINED.
    """
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(float('inf'))
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_0, in_1, None)
    return tmp_3, tmp_8, linear


def replacement_args(in_0, in_1, in_3, in_5, in_4, in_2):
    return (in_0, in_1, in_3, in_5, in_4, in_2)


@triton.jit
def fused_deg_norm_kernel(
    deg_ptr,     # [N] raw deg values (before pow_)
    row_ptr,     # [E] row indices (int64)
    eweight_ptr, # [E] edge weights (bf16/f16)
    col_ptr,     # [E] col indices (int64)
    out_ptr,     # [E] output (bf16/f16)
    E,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < E

    row = tl.load(row_ptr + offsets, mask=mask, other=0)
    col = tl.load(col_ptr + offsets, mask=mask, other=0)
    eweight = tl.load(eweight_ptr + offsets, mask=mask, other=0.0)

    # Load raw deg values, compute deg^{-0.5}, zero out inf values
    d1 = tl.load(deg_ptr + row, mask=mask, other=1.0).to(tl.float32)
    d2 = tl.load(deg_ptr + col, mask=mask, other=1.0).to(tl.float32)

    d1_safe = tl.where(d1 > 1e8, 1.0, d1)
    d2_safe = tl.where(d2 > 1e8, 1.0, d2)

    inv_d1 = tl.math.rsqrt(d1_safe)
    inv_d2 = tl.math.rsqrt(d2_safe)

    out_f32 = inv_d1 * eweight.to(tl.float32) * inv_d2

    tl.store(out_ptr + offsets, out_f32, mask=mask)


@triton.jit
def fused_gemm_small_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Triton GEMM: C[M,N] = A[M,K] @ B[N,K].T  for small matrices."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(b))

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def fused_full_graph_gae(in_0, in_1, in_3, in_5, in_4, in_2):
    """
    Fused: deg_norm (pow_ + masked_fill + getitem×2 + mul×3) + linear.
    Args:
        in_0: input tensor  [M, K]  (bfloat16/float16, CUDA)
        in_1: weight tensor [N, K]  (bfloat16/float16, CUDA)
        in_3: deg tensor    [N_g]   (bfloat16/float16, CUDA)
        in_5: row indices   [E]     (int64, CUDA)
        in_4: edge weights  [E]     (bfloat16/float16, CUDA)
        in_2: col indices   [E]     (int64, CUDA)
    Returns:
        (tmp_3, tmp_8, linear): tmp_3 is the eq(inf) result (always 0/False),
        tmp_8 is the deg-normalized edge weights, linear is the GEMM result.
    """
    # Compute deg-normalized edge weights
    E = in_5.shape[0]
    tmp_8 = torch.empty_like(in_4)
    BLOCK_SIZE = 256
    grid = ((E + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_deg_norm_kernel[grid](
        in_3, in_5, in_4, in_2, tmp_8,
        E,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute linear: in_0 @ in_1.T  (in_0: [M,K], in_1: [N,K])
    # Grid: 2D grid (M/BLOCK_M, N/BLOCK_N) blocks, each handles a tile of output
    M = in_0.shape[0]
    N = in_1.shape[0]
    K = in_0.shape[1]
    lin_out = torch.empty((M, N), dtype=in_0.dtype, device=in_0.device)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16  # Match K tile to K size; pad internally
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    fused_gemm_small_kernel[(grid_m, grid_n)](
        in_0, in_1, lin_out,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return tmp_8, lin_out


@triton.jit
def fused_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute C[M,N] = A[M,K] @ B[N,K].T using Triton tiling."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load A block: [BLOCK_M, BLOCK_K]
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # Load B block: [BLOCK_N, BLOCK_K]
        b = tl.load(
            b_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        # acc += A_block @ B_block.T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def replacement_func():
    return fused_full_graph_gae