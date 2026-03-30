import torch
import triton
import triton.language as tl

def pattern(arg1, arg2, arg3):
    # Create a chain of operations that matches the structure of the computation
    tmp1 = arg1 + arg2
    tmp2 = tmp1 * arg3
    return tmp2

def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

@triton.jit
def linear_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_N = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_M
    pid_n = pid // num_pid_M

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = in_ptr + (offs_m[:, None] * stride + offs_k[None, :])
    b_ptrs = weight_ptr + (offs_n[None, :] * K + offs_k[:, None])
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M)[:, None] & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N)[None, :] & (offs_k[:, None] < K), other=0.0)
        accumulator += tl.dot(a, b, trans_b=True)
        a_ptrs += BLOCK_SIZE_K * stride
        b_ptrs += BLOCK_SIZE_K * N
    
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    accumulator = accumulator + bias[None, :]
    
    out_ptrs = out_ptr + (offs_m[:, None] * stride + offs_n[None, :])
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < M)[:, None] & (offs_n[None, :] < N))

@torch.fx.wrap
def triton_linear(in_6, in_5, in_4):
    M, K = in_6.shape
    N = in_5.shape[0]
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    num_pid_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_N = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (num_pid_M * num_pid_N,)
    
    out = torch.empty((M, N), device=in_6.device, dtype=in_6.dtype)
    
    linear_kernel[grid](
        in_6,
        in_5,
        in_4,
        out,
        in_6.stride(1),
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return triton_linear