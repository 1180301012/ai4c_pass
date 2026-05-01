import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + (block_start_m + tl.arange(0, BLOCK_M)) * K + k,
            mask=(block_start_m + tl.arange(0, BLOCK_M)) < M,
            other=0.0
        )
        b = tl.load(
            B_ptr + k + (tl.arange(0, BLOCK_K) * N) + block_start_n,
            mask=(k + tl.arange(0, BLOCK_K)) < K,
            other=0.0
        )
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + (block_start_m + tl.arange(0, BLOCK_M)) * N + block_start_n,
        acc,
        mask=(block_start_m + tl.arange(0, BLOCK_M)) < M
    )

@torch.fx.wrap
def conv1x1_to_matmul(input, weight, bias):
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    M = N * H * W

    input_flat = input.permute(0, 2, 3, 1).reshape(M, C_in)
    weight_reshaped = weight.permute(0, 2, 3, 1).view(C_out, C_in)
    output_flat = torch.empty(M, C_out, device=input.device, dtype=input.dtype)

    BLOCK_M = 256
    BLOCK_N = 16
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))
    matmul_kernel[grid](
        input_flat,
        weight_reshaped,
        output_flat,
        M,
        C_out,
        C_in,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )

    output_flat += bias.view(1, C_out)
    output = output_flat.view(N, C_out, H, W)
    return output

def pattern(input, weight, bias):
    return torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(input, weight, bias):
    return (input, weight, bias)

def replacement_func():
    return conv1x1_to_matmul