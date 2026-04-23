import torch
import triton
import triton.language as tl

# Pattern matching - only match matmul + scalar mul (not transpose)
# The transpose operation will remain in the graph
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1

# Argument extraction
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Fused kernel: matmul + scalar mul
# Computes: out = in_0 * (in_2 @ in_1)  [M, N]
@triton.jit
def fused_matmul_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    M, K, N,
    stride_in1_0, stride_in1_1,
    stride_in2_0, stride_in2_1,
    stride_out_0, stride_out_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load the scalar multiplier
    scale = tl.load(in_0_ptr)

    # Compute offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load in_2 tile: [BLOCK_M, BLOCK_K]
        a_ptrs = in_2_ptr + offs_m[:, None] * stride_in2_0 + offs_k[None, :] * stride_in2_1
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load in_1 tile: [BLOCK_K, BLOCK_N]
        b_ptrs = in_1_ptr + offs_k[:, None] * stride_in1_0 + offs_n[None, :] * stride_in1_1
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate matrix multiplication
        acc += tl.dot(a, b, allow_tf32=False)

    # Multiply by scalar
    acc = acc * scale

    # Store out [M, N] - the scaled matmul result
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_0 + offs_n[None, :] * stride_out_1
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_matmul_mul(in_0, in_1, in_2):
    # in_0: scalar, in_1: [K, N], in_2: [M, K]
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]

    # Output tensor
    out = torch.empty((M, N), dtype=in_1.dtype, device=in_1.device)

    # Block sizes
    BLOCK_M = min(M, 16)
    BLOCK_N = min(N, 16)
    BLOCK_K = 64

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    fused_matmul_mul_kernel[grid](
        in_0, in_1, in_2, out,
        M, K, N,
        in_1.stride(0), in_1.stride(1),
        in_2.stride(0), in_2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out


def replacement_func():
    return fused_matmul_mul