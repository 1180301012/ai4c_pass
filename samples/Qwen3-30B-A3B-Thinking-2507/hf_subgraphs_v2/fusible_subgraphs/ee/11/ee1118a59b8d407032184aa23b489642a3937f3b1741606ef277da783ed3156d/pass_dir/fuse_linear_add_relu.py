import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused linear + addition + ReLU
def get_block_sizes(M, N, K):
    # Tune block sizes based on problem size
    # We'll use 64x128 blocks for good occupancy
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    return BLOCK_M, BLOCK_N, BLOCK_K

@triton.jit
def fused_linear_add_relu_kernel(
    A,  # [M, K]
    B,  # [K, N]
    bias,  # [N]
    residual,  # [M, N]
    out,  # [M, N]
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Define thread offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    m_offset = block_start_m + tl.arange(0, BLOCK_M)
    n_offset = block_start_n + tl.arange(0, BLOCK_N)

    mask_m = m_offset < M
    mask_n = n_offset < N

    # Initialize accumulator to 0.0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K blocks
    for k in range(0, K, BLOCK_K):
        # Load A block: [BLOCK_M, BLOCK_K]
        a = tl.load(
            A + m_offset[:, None] * K + (k + tl.arange(0, BLOCK_K)),
            mask=(mask_m[:, None] & (k + tl.arange(0, BLOCK_K) < K)),
            other=0.0
        )
        # Load B block: [BLOCK_K, BLOCK_N]
        b = tl.load(
            B + (k + tl.arange(0, BLOCK_K))[:, None] * N + n_offset,
            mask=((k + tl.arange(0, BLOCK_K))[:, None] < K) & mask_n[None, :],
            other=0.0
        )
        # Accumulate product
        acc += tl.dot(a, b)

    # Load bias: [BLOCK_N]
    bias_vals = tl.load(bias + n_offset, mask=mask_n)
    # Load residual: [BLOCK_M, BLOCK_N]
    residual_vals = tl.load(
        residual + m_offset[:, None] * N + n_offset,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    )
    # Combine and apply ReLU
    out_vals = acc + bias_vals[None, :] + residual_vals
    out_vals = tl.maximum(0.0, out_vals)
    # Store result
    tl.store(
        out + m_offset[:, None] * N + n_offset,
        out_vals,
        mask=mask_m[:, None] & mask_n[None, :]
    )

# Kernel wrapper
@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    # Get dimensions
    M, K = in_3.shape
    K_in, N = in_1.shape
    assert K == K_in, f"Incompatible shapes: input {in_3.shape}, weight {in_1.shape}"
    
    # Create output tensor
    out = torch.empty_like(in_3)

    # Set optimal block sizes for this problem
    BLOCK_M, BLOCK_N, BLOCK_K = get_block_sizes(M, N, K)

    # Compute grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)

    # Launch kernel
    fused_linear_add_relu_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return out

# Replacement function

def replacement_func():
    return fused_linear_add_relu