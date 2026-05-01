import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_add_relu_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    in_2_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    row_offsets = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    row_mask = row_offsets < M
    col_mask = col_offsets < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        a = tl.load(
            in_3_ptr + row_offsets[:, None] * K + k_offsets[None, :]
            , mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        b = tl.load(
            in_1_ptr + k_offsets[:, None] * N + col_offsets[None, :]
            , mask=k_mask[:, None] & col_mask[None, :], other=0.0)

        acc += a * b

    bias = tl.load(in_0_ptr + col_offsets, mask=col_mask, other=0.0)
    bias = bias[None, :]  
    acc += bias

    residual = tl.load(
        in_2_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        , mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    acc += residual

    acc = tl.maximum(acc, 0.0)

    tl.store(
        out_ptr + row_offsets[:, None] * N + col_offsets[None, :]
        , acc, mask=row_mask[:, None] & col_mask[None, :]
    )

@torch.fx.wrap
def fused_linear_add_relu(x, weight, bias, residual):
    M, K = x.shape
    N = weight.shape[1]
    out = torch.empty_like(x)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    fused_linear_add_relu_kernel[(grid_m, grid_n)](
        x, weight, bias, residual, out,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out

def pattern(in_3, in_1, in_0, in_2):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4

def replacement_args(in_3, in_1, in_0, in_2):
    return (in_3, in_1, in_0, in_2)

def replacement_func():
    return fused_linear_add_relu