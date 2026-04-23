import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a_row, stride_b_row, stride_c_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_end = min(k + BLOCK_K, K)
        offs_k_inner = offs_k[:k_end - k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_a_row + offs_k_inner[None, :])
        b_ptrs = B_ptr + (offs_k_inner[:, None] * stride_b_row + offs_n[None, :])
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k_inner[None, :] < k_end), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k_inner[:, None] < k_end) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c = accumulator
    c_ptrs = C_ptr + (offs_m[:, None] * stride_c_row + offs_n[None, :])
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@torch.fx.wrap
def optimized_conv2d(in_0, in_1, in_2):
    # in_0: bias [N], in_1: weight [N, C, 1, 1], in_2: input [B, C, H, W]
    B, C, H, W = in_2.shape
    N = in_1.shape[0]
    K = C
    M = B * H * W

    # Reshape input to [M, K]
    in_2_flat = in_2.view(M, K)
    weight_flat = in_1.view(N, K)

    # Initialize output
    output = torch.empty(M, N, device=in_2.device, dtype=in_2.dtype)
    output = output.view(B, N, H, W)

    # Config
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Call kernel
    gemm_kernel[grid](
        in_2_flat, weight_flat, output,
        M, N, K,
        in_2_flat.stride(0), weight_flat.stride(0), output.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Add bias if present
    if in_0 is not None:
        output = output + in_0.view(1, N, 1, 1)

    return output

def pattern(in_0, in_1, in_2):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp3 = torch.stack([conv], dim=0)
    tmp4 = tmp3.sum(dim=0)
    return tmp4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return optimized_conv2d