import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: 1x1 conv2d -> silu -> dropout(p=0, training=False)
    in_0 = bias  [Cout]
    in_1 = weight [Cout, Cin, 1, 1]
    in_2 = input  [N, Cin, H, W]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: GEMM (C[N,M] = A[N,K] @ B[K,M]) + bias broadcast + SiLU
#   A = weight  [Cout=N, Cin=K]  row-major     stride_n=K,  stride_k=1
#   B = input   [Cin=K, H*W=M]  col-major     stride_k=H*W, stride_m=1
#   C = output  [Cout=N, H*W=M] col-major     stride_n=H*W, stride_m=1
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_silu_kernel(
    w_ptr,      # weight  [N, K]  row-major
    in_ptr,     # input   [K, M]  col-major
    bias_ptr,   # bias    [N]
    out_ptr,    # output  [N, M]  col-major
    M, N, K,
    stride_an,  # weight N-stride  = K
    stride_ak,  # weight K-stride  = 1
    stride_bk,  # input  K-stride  = H*W
    stride_bm,  # input  M-stride  = 1
    stride_cn,  # output N-stride  = H*W
    stride_cm,  # output M-stride  = 1
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)   # tile along Cout dimension
    pid_m = tl.program_id(1)   # tile along spatial dimension

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for A tile [BLOCK_N, BLOCK_K] and B tile [BLOCK_K, BLOCK_M]
    a_ptrs = w_ptr  + offs_n[:, None] * stride_an + offs_k[None, :] * stride_ak
    b_ptrs = in_ptr + offs_k[:, None] * stride_bk + offs_m[None, :] * stride_bm

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_iter * BLOCK_K
        mask_a = (offs_n[:, None] < N) & ((k_off + offs_k)[None, :] < K)
        mask_b = ((k_off + offs_k)[:, None] < K) & (offs_m[None, :] < M)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcasted over M dimension)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[:, None].to(tl.float32)

    # Fused SiLU activation: x * sigmoid(x)
    silu = acc * tl.sigmoid(acc)

    # Write result back in col-major layout  [N, M]
    c_ptrs = out_ptr + offs_n[:, None] * stride_cn + offs_m[None, :] * stride_cm
    mask_c = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(c_ptrs, silu.to(out_ptr.dtype.element_ty), mask=mask_c)


@torch.fx.wrap
def conv1x1_silu(in_0, in_1, in_2):
    """
    in_0 : bias   [Cout]
    in_1 : weight [Cout, Cin, 1, 1]
    in_2 : input  [N_batch, Cin, H, W]  (N_batch == 1)
    """
    N_batch, Cin, H, W = in_2.shape
    Cout = in_1.shape[0]

    # GEMM dimensions
    M = H * W      # spatial = 1024
    N = Cout       # 256
    K = Cin        # 128

    out = torch.empty((N_batch, Cout, H, W), dtype=in_2.dtype, device=in_2.device)

    # Weight A [N, K] row-major: stride_n = K = Cin, stride_k = 1
    stride_an = in_1.stride(0)   # = K = Cin
    stride_ak = in_1.stride(1)   # = 1

    # Input  B [K, M] – stored as NCHW with N=1:
    #   B[k, m] = in_2[0, k, h, w]  addr = k*H*W + m
    #   stride_k = H*W = in_2.stride(1),  stride_m = 1 = in_2.stride(3)
    stride_bk = in_2.stride(1)   # = H*W
    stride_bm = in_2.stride(3)   # = 1

    # Output C [N, M] – stored as NCHW with N=1:
    #   C[n, m] = out[0, n, h, w]   addr = n*H*W + m
    stride_cn = out.stride(1)    # = H*W
    stride_cm = out.stride(3)    # = 1

    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(M, meta['BLOCK_M']),
    )

    _conv1x1_silu_kernel[grid](
        in_1, in_2, in_0, out,
        M, N, K,
        stride_an, stride_ak,
        stride_bk, stride_bm,
        stride_cn, stride_cm,
    )

    return (out,)


def replacement_func():
    return conv1x1_silu