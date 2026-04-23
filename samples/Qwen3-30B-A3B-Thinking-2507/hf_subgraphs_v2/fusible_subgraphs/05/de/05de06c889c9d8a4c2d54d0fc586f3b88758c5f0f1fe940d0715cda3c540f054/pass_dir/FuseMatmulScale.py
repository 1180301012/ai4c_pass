import torch
import triton
import triton.language as tl

@triton.jit

def matmul_scaled_kernel(
    a_ptr, b_ptr, c_ptr, s,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M) & (offs_n < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_ak = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_ak[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_ak[None, :] < K),
            other=0.0
        )
        offs_bk = k + tl.arange(0, BLOCK_K)
        b = tl.load(
            b_ptr + offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_bk[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)
    
    c = acc * s
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask
    )

@torch.fx.wrap

def custom_matmul_scaled(a, b, s):
    M, K = a.shape
    K, N = b.shape
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)
    
    s_val = s.item()
    
    grid = (M, N)
    
    matmul_scaled_kernel[grid](
        a, b, out, s_val,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return out

def pattern(in_2, in_1, in_0):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    return custom_matmul_scaled