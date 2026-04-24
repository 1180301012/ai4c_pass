import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A @ B where A is [M,K] (weights) and B is [K,N] (input)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_conv1x1(in_0, in_1):
    """
    1x1 conv replacement: computes [1, C_out, H, W] using Triton GEMM.
    in_0: weight [C_out, C_in, 1, 1]
    in_1: input  [1, C_in, H, W]
    Returns: [1, C_out, H, W]  (unfold+reshape handled by remaining graph)
    """
    C_out = in_0.shape[0]
    C_in  = in_0.shape[1]
    H     = in_1.shape[2]
    W     = in_1.shape[3]
    M, N, K = C_out, H * W, C_in   # 128, 1024, 256

    # Allocate output [1, C_out, H, W] — factory function, no aten.view needed
    out = torch.empty((1, C_out, H, W), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    # weight [C_out,C_in,1,1]: stride_m=C_in, stride_k=H*W
    # input  [1,C_in,H,W]:    stride_k=H*W, stride_n=1
    # output [1,C_out,H,W]:   stride_m=H*W, stride_n=1
    _conv1x1_matmul_kernel[grid](
        in_0, in_1, out,
        M, N, K,
        K, 1,       # stride_am, stride_ak
        N, 1,       # stride_bk, stride_bn
        N, 1,       # stride_cm, stride_cn
    )
    return out


def pattern(in_0, in_1):
    return torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_conv1x1