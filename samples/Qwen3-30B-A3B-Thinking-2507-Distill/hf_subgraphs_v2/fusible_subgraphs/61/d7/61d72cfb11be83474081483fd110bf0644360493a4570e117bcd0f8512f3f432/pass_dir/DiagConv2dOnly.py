import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def diag_conv2d(in_0, in_1):
    # in_0: weight [128, 256, 1, 1]
    # in_1: input  [1, 256, 32, 32]
    # output: [1, 128, 32, 32]
    C_out = in_0.shape[0]
    C_in  = in_0.shape[1]
    H     = in_1.shape[2]
    W     = in_1.shape[3]
    M, N, K = C_out, H * W, C_in   # 128, 1024, 256
    out = torch.empty((1, C_out, H, W), dtype=in_1.dtype, device=in_1.device)
    # Reshape for matmul: a=[C_out, C_in], b=[C_in, H*W]
    a = in_0.reshape(C_out, C_in)      # [128, 256]
    b = in_1.reshape(1, C_in, H * W)   # [1, 256, 1024]
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # Need to copy b to [C_in, H*W] and a to [C_out, C_in] contiguous
    # Then do batched matmul
    # Simplest: reshape b to [C_in, H*W], a to [C_out, C_in]
    # The output will be [C_out, H*W] which is reshaped to [1, C_out, H, W]
    import ctypes
    b_flat = b.reshape(C_in, H * W)           # [256, 1024] contiguous
    a_flat = a.reshape(C_out, C_in)            # [128, 256] contiguous
    c_flat = torch.empty((C_out, H * W), dtype=in_1.dtype, device=in_1.device)
    _matmul_kernel[grid](
        a_flat, b_flat, c_flat,
        M, N, K,
        a_flat.stride(0), a_flat.stride(1),
        b_flat.stride(0), b_flat.stride(1),
        c_flat.stride(0), c_flat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c_flat.reshape(1, C_out, H, W)


def pattern(in_0, in_1):
    return torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return diag_conv2d