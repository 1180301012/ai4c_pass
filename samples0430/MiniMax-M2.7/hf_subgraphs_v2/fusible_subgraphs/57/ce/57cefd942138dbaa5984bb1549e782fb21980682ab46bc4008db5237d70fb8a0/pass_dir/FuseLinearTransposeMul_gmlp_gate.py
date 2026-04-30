import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_transpose_mul_kernel(
    x_ptr,  # [B, M, K]
    W_ptr,  # [K, N] (row-major in memory, K contiguous rows)
    bias_ptr,  # [N]
    u_ptr,  # [B, N, M]
    out_ptr,  # [B, M, N]
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel for: linear(x, W, bias) -> transpose -> multiply with u
    This fuses:
    1. Linear: x @ W^T + bias  [B, M, K] x [K, N] -> [B, M, N]
    2. Transpose: [B, M, N] -> [B, N, M] (conceptually)
    3. Multiply: result * u (element-wise, broadcasting u to [B, N, M])

    Note: The "transpose" is implicit - we compute the linear result and directly
    multiply with u which has complementary shape [B, N, M].
    """
    # Program layout: pid(0) = b*n (batch*col), pid(1) = row
    # Output element: out[b, row, col] where row in [0,M), col in [0,N)
    b = (tl.program_id(0) // N).to(tl.int32)
    col = (tl.program_id(0) % N).to(tl.int32)
    row = tl.program_id(1)

    # Compute pointers for this output position
    # out[b, row, col]
    x_row_offset = b * M * K + row * K
    u_offset = b * N * M + col * M + row  # u[b, col, row]

    # Initialize accumulator for dot product x[row] @ W[:, col]
    acc = 0.0

    # Load bias for this column
    bias = tl.load(bias_ptr + col)

    # Loop over K dimension with vectorized loads
    for k in range(0, K, BLOCK_N):
        # Vectorized load of x row: x[b, row, k:k+BLOCK_N]
        x_offsets = x_row_offset + tl.arange(0, BLOCK_N)
        x_mask = x_offsets < b * M * K + M * K
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # Load W column: W[k:k+BLOCK_N, col] (column access)
        W_offsets = k * N + col + tl.arange(0, BLOCK_N) * N
        W_mask = W_offsets < K * N
        W_vals = tl.load(W_ptr + W_offsets, mask=W_mask, other=0.0)

        # Multiply accumulate
        acc += tl.sum(x_vals * W_vals)

    # Add bias
    acc = acc + bias

    # Apply compute function and multiply with u[b, col, row]
    u_val = tl.load(u_ptr + u_offset)
    out_val = acc * u_val  # Element-wise multiplication

    # Store result
    out_offset = b * M * N + row * N + col
    tl.store(out_ptr + out_offset, out_val)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match: linear(in_2, in_1, in_0) -> transpose(-1,-2) -> multiply with in_3
    This is the gMLP gating pattern.
    
    Args:
        in_0: bias [N]
        in_1: weight [K, N]
        in_2: input [B, M, K]
        in_3: u [B, N, M]
    
    Returns:
        (linear_result, output)
        - linear_result: intermediate for potential reuse
        - output: final result [B, N, M]
    """
    linear_result = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear_result.transpose(-1, -2)
    output = in_3 * tmp_3
    return linear_result, output


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    Order: (x, W, bias, u) matching the kernel signature.
    """
    return (in_2, in_1, in_0, in_3)


@torch.fx.wrap
def kernel_wrapper(x, W, bias, u):
    """
    Wrapper for the fused kernel.
    
    x: [B, M, K]
    W: [K, N]
    bias: [N]
    u: [B, N, M]
    
    Returns: [B, N, M] - element-wise product of linear result and u
    """
    B, M, K = x.shape
    N = bias.shape[0]

    # Grid: (B*N, M) - one program per output element
    grid = (B * N, M)

    # Block sizes
    BLOCK_M = 1  # One row per program
    BLOCK_N = 64  # Vectorized load size for K dimension

    # Output: [B, M, N]
    out = torch.empty((B, M, N), dtype=x.dtype, device=x.device)

    fused_linear_transpose_mul_kernel[grid](
        x, W, bias, u, out,
        B=B, M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    return out


def replacement_func():
    """
    Returns the replacement function for the fused linear-transpose-mul operation.
    """
    return kernel_wrapper