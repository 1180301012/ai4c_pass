import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3, in_5, in_4, in_2):
    """
    Full-graph pattern for RECT_L: deg_norm chain + linear(in_1, in_0, None).
    Note: RECT_L has linear arguments SWAPPED compared to GAE.
    in_3 is a pattern PLACEHOLDER (exempt from NOT_CONTAINED).
    """
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(float('inf'))
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_1, in_0, None)
    return tmp_3, tmp_8, linear


def replacement_args(in_0, in_1, in_3, in_5, in_4, in_2):
    return (in_0, in_1, in_3, in_5, in_4, in_2)


@triton.jit
def fused_deg_norm_kernel_r(
    deg_ptr,
    row_ptr,
    eweight_ptr,
    col_ptr,
    out_ptr,
    E,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < E

    row = tl.load(row_ptr + offsets, mask=mask, other=0)
    col = tl.load(col_ptr + offsets, mask=mask, other=0)
    eweight = tl.load(eweight_ptr + offsets, mask=mask, other=0.0)

    d1 = tl.load(deg_ptr + row, mask=mask, other=1.0).to(tl.float32)
    d2 = tl.load(deg_ptr + col, mask=mask, other=1.0).to(tl.float32)

    d1_safe = tl.where(d1 > 1e8, 1.0, d1)
    d2_safe = tl.where(d2 > 1e8, 1.0, d2)

    inv_d1 = tl.math.rsqrt(d1_safe)
    inv_d2 = tl.math.rsqrt(d2_safe)

    out_f32 = inv_d1 * eweight.to(tl.float32) * inv_d2

    tl.store(out_ptr + offsets, out_f32, mask=mask)


@triton.jit
def fused_gemm_small_kernel_r(
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
def fused_full_graph_rect(in_0, in_1, in_3, in_5, in_4, in_2):
    """
    Fused: deg_norm + linear(in_1, in_0, None).
    Args:
        in_0: x tensor    [M, K]
        in_1: weight tensor [N, K]
        in_3: deg tensor    [N_g]
        in_5: row indices   [E]
        in_4: edge weights  [E]
        in_2: col indices   [E]
    Returns:
        (tmp_8, linear) — tmp_3 is dead code (eq result), omitted
    """
    # deg-normalized edge weights
    E = in_5.shape[0]
    tmp_8 = torch.empty_like(in_4)
    BLOCK_SIZE = 256
    grid = ((E + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_deg_norm_kernel_r[grid](
        in_3, in_5, in_4, in_2, tmp_8,
        E,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # linear: in_1 @ in_0.T  (weight[N,K] @ input[K,M] = [N,M])
    M = in_0.shape[0]
    N = in_1.shape[0]
    K = in_0.shape[1]
    lin_out = torch.empty((N, M), dtype=in_0.dtype, device=in_0.device)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    fused_gemm_small_kernel_r[(grid_m, grid_n)](
        in_1, in_0, lin_out,
        N, M, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return tmp_8, lin_out


def replacement_func():
    return fused_full_graph_rect