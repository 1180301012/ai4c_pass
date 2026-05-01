import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_12, in_4):
    softmax_result = tmp_12.softmax(dim=-1)
    matmul_result = softmax_result @ in_4
    return softmax_result, matmul_result

# Argument extraction function
def replacement_args(tmp_12, in_4):
    return (tmp_12, in_4)

# Triton kernel for fused softmax + matmul
@triton.jit
def fused_softmax_matmul_kernel(
    A_ptr, A_batch_stride, A_row_stride, A_col_stride,
    B_ptr, B_batch_stride, B_col_stride, B_out_stride,
    C_ptr, C_batch_stride, C_row_stride, C_out_stride,
    batch_size, row_size, col_size, out_col_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Calculate block IDs
    batch_id = tl.program_id(0)
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)

    # Calculate start indices for the current tile
    m_start = block_m * BLOCK_M
    n_start = block_n * BLOCK_N

    # Allocate registers for the output tile
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute the sum for each row in the current tile
    row_sums = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Accumulate exp(A) * B over blocks of cols
    for k_start in range(0, col_size, BLOCK_K):
        # Load a tile of A: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + batch_id * A_batch_stride + m_start * A_row_stride + k_start * A_col_stride,
            shape=(BLOCK_M, BLOCK_K),
            mask=(m_start + tl.arange(0, BLOCK_M) < row_size) & (k_start + tl.arange(0, BLOCK_K) < col_size)
        )
        # Compute exp
        a_exp = tl.exp(a)
        # Load a tile of B: [BLOCK_K, BLOCK_N]
        b = tl.load(
            B_ptr + batch_id * B_batch_stride + k_start * B_col_stride + n_start * B_out_stride,
            shape=(BLOCK_K, BLOCK_N),
            mask=(k_start + tl.arange(0, BLOCK_K) < col_size) & (n_start + tl.arange(0, BLOCK_N) < out_col_size)
        )
        # Accumulate a_exp * b
        accumulator += tl.dot(a_exp, b)
        # Accumulate the sum for each row
        row_sums += tl.sum(a_exp, axis=1)

    # Divide accumulator by row_sums
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            if m_start + m < row_size and n_start + n < out_col_size:
                accumulator[m, n] = accumulator[m, n] / row_sums[m]

    # Store the result
    tl.store(
        C_ptr + batch_id * C_batch_stride + m_start * C_row_stride + n_start * C_out_stride,
        accumulator,
        mask=(m_start + tl.arange(0, BLOCK_M) < row_size) & (n_start + tl.arange(0, BLOCK_N) < out_col_size)
    )

# Kernel wrapper
@torch.fx.wrap
def fused_softmax_matmul(tmp_12, in_4):
    batch, row, col = tmp_12.shape
    out_col = in_4.shape[2]

    # Create output tensor
    out = torch.empty_like(tmp_12)

    # Calculate strides (row-major layout)
    A_batch_stride = row * col
    A_row_stride = col
    A_col_stride = 1

    B_batch_stride = col * out_col
    B_col_stride = out_col
    B_out_stride = 1

    # Set kernel block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    # Compute grid dimensions
    grid_m = (row + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_col + BLOCK_N - 1) // BLOCK_N
    grid = (batch, grid_m, grid_n)

    # Launch kernel
    fused_softmax_matmul_kernel[grid](
        tmp_12,
        A_batch_stride,
        A_row_stride,
        A_col_stride,
        in_4,
        B_batch_stride,
        B_col_stride,
        B_out_stride,
        out,
        batch * row * out_col,
        row * out_col,
        out_col,
        batch,
        row,
        col,
        out_col,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    return out

# Replacement function (returns kernel wrapper)
def replacement_func():
    return fused_softmax_matmul