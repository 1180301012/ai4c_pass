import torch
import triton
import triton.language as tl


def pattern(in_2, in_1, in_0, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4

def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)


@triton.jit
def fused_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    in_3_ptr,
    out_ptr,
    batch_size,
    rows,
    cols,
    out_features,
    output_rows,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    batch_id = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)
    
    m_start = m_block * BLOCK_M
    n_start = n_block * BLOCK_N

    # Load bias for current block
    bias = tl.load(B_ptr + m_start, 
                  mask=m_start + tl.arange(0, BLOCK_M) < out_features,
                  other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matrix multiplication over k dimension
    for k_start in range(0, cols, BLOCK_K):
        # Load W block: [BLOCK_M, BLOCK_K]
        w_block = tl.load(
            W_ptr + m_start * cols + k_start,
            shape=(BLOCK_M, BLOCK_K),
            mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < out_features) & 
                  (k_start + tl.arange(0, BLOCK_K)[None, :] < cols),
            other=0.0
        )

        # Load X block: [BLOCK_K, BLOCK_N] (X is [batch, rows, cols])
        x_block = tl.load(
            X_ptr + batch_id * rows * cols + n_start * cols + k_start,
            shape=(BLOCK_K, BLOCK_N),
            mask=(k_start + tl.arange(0, BLOCK_K)[:, None] < cols) & 
                  (n_start + tl.arange(0, BLOCK_N)[None, :] < rows),
            other=0.0
        )

        # Accumulate dot product
        acc += tl.dot(w_block, x_block)

    # Apply bias and element-wise multiply
    acc = acc + bias[:, None]
    in_3_block = tl.load(
        in_3_ptr + batch_id * out_features * output_rows + m_start * output_rows + n_start,
        shape=(BLOCK_M, BLOCK_N),
        mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < out_features) & 
              (n_start + tl.arange(0, BLOCK_N)[None, :] < output_rows),
        other=0.0
    )
    out_block = acc * in_3_block

    # Store result
    tl.store(
        out_ptr + batch_id * out_features * output_rows + m_start * output_rows + n_start,
        out_block,
        mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < out_features) & 
              (n_start + tl.arange(0, BLOCK_N)[None, :] < output_rows)
    )


@torch.fx.wrap
def fuse_linear_transpose_multiply(X, W, B, in_3):
    batch_size, rows, cols = X.shape
    out_features = W.shape[0]
    output_rows = in_3.shape[2]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Calculate grid dimensions
    num_m_blocks = (out_features + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (output_rows + BLOCK_N - 1) // BLOCK_N
    grid = (batch_size, num_m_blocks, num_n_blocks)

    # Create output tensor
    out = torch.empty_like(in_3)

    # Launch kernel
    fused_kernel[grid](
        X,
        W,
        B,
        in_3,
        out,
        batch_size,
        rows,
        cols,
        out_features,
        output_rows,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )

    return out

def replacement_func():
    return fuse_linear_transpose_multiply