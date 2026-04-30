import torch
import triton
import triton.language as tl


@triton.jit
def edge_norm_kernel(
    norm_ptr, row_ptr, col_ptr, edge_weight_ptr, out_ptr,
    num_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    row_idx = tl.load(row_ptr + offsets, mask=mask)
    col_idx = tl.load(col_ptr + offsets, mask=mask)

    norm_row = tl.load(norm_ptr + row_idx, mask=mask)
    norm_col = tl.load(norm_ptr + col_idx, mask=mask)

    ew = tl.load(edge_weight_ptr + offsets, mask=mask)

    result = norm_row * ew * norm_col

    tl.store(out_ptr + offsets, result, mask=mask)


def _do_edge_norm(norm, row, col, edge_weight):
    num_edges = row.shape[0]
    out = torch.empty_like(edge_weight)
    BLOCK_SIZE = 2048
    grid = ((num_edges + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    edge_norm_kernel[grid](
        norm, row, col, edge_weight, out,
        num_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_x, other=0.0
        )
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=mask_w, other=0.0
        )
        acc += tl.dot(x_tile, w_tile)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_out)


def _do_linear(x, w):
    M = x.shape[0]
    K = x.shape[1]
    N = w.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if N <= 16:
        BLOCK_M = 64
        BLOCK_N = 16
        BLOCK_K = 32
    else:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
    grid = (((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N),)
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


@torch.fx.wrap
def dispatch(a, b, c, d, route):
    if route == "edge_norm":
        return _do_edge_norm(a, b, c, d)
    elif route == "linear":
        return _do_linear(a, b)
    return a