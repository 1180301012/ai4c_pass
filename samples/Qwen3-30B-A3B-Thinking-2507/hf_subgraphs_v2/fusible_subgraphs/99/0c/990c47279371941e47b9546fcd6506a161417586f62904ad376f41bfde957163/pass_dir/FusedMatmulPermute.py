import torch
import triton
import triton.language as tl

@triton.jit
def matmul_permute_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)
    
    a = tl.load(
        a_ptr + pid_b * M * K + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    
    b = tl.load(
        b_ptr + pid_b * K * N + offs_k[:, None] * N + offs_n[None, :],
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
    
    acc = tl.dot(a, b)
    
    tl.store(
        c_ptr + pid_b * N * M + offs_n[:, None] * M + offs_m[None, :],
        acc,
        mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
    )

@torch.fx.wrap
def fused_matmul_permute(tmp_1, in_1):
    B, M, K = tmp_1.shape
    N = in_1.shape[2]
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    num_programs_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    out = torch.empty((B, N, M), dtype=tmp_1.dtype, device=tmp_1.device)
    matmul_permute_kernel[(B, num_programs_m, num_programs_n)](
        tmp_1, in_1, out, B, M, K, N, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return out

def pattern(tmp_1, in_1):
    matmul = torch.matmul(tmp_1, in_1)
    output = matmul.permute(0, 2, 1)
    return output

def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)

def replacement_func():
    return fused_matmul_permute