import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d  +  unfold(kernel=2x2, stride=2x2)  +  reshape(1,128,4,-1)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton GEMM kernel:  strides-2 gather of in_1  +  matrix product  +  reshape
#
# Mathematically equivalent to:
#   A[m, k] = in_1[0, k, m//16, m%16]    (M=8192, K=256)
#   B[k, n] = in_0[n, k]                  (K=256, N=128)
#   C = A @ B                            (→ already stored as [1,128,4,4])
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 4}),
    ],
    key=[],
)
@triton.jit
def fused_conv_unfold_reshape_kernel(
    a_ptr,      # in_1 base pointer  [1, 256, 32, 32] — treated as [M, K]
    b_ptr,      # in_0 base pointer  [128, 256, 1, 1] — treated as [K, N]
    c_ptr,      # output pointer     [1, 128, 4, 4]  — treated as [M, N]
    M: tl.constexpr,        # 8192  (16 * 16 = 256 patches)
    N: tl.constexpr,        # 128   (output channels)
    K: tl.constexpr,        # 256   (input channels)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ---- Input A: in_1[0, k, ih, iw]  →  A[m, k] --------------------------------
    # m = ih * 16 + iw  (16 × 16 patch grid from unfold)
    # ih = m // 16,  iw = m % 16
    # A[m, k] = a_ptr[ k * 1024 + ih * 32 + iw ]
    #          = a_ptr[ k * 1024 + (m // 16) * 32 + (m % 16) ]
    offs_k = tl.arange(0, BLOCK_K)

    ih = offs_m // 16          # row of 16×16 patch grid
    iw = offs_m % 16           # col of 16×16 patch grid

    # pointer tile: [BLOCK_M, BLOCK_K]
    a_base = offs_k * 1024     # 1024 = 32 * 32 (in_1 spatial size)
    a_ptr_tile = a_ptr + a_base[None, :] + ih[:, None] * 32 + iw[:, None]

    # ---- Weight B: in_0[n, k]  →  B[k, n] ----------------------------------------
    # B stored as [N, K] in memory, so B[k, n] = b_ptr[ n * K + k ]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k

        # A tile [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptr_tile,
                    mask=(offs_m[:, None] < M) & (k[None, :] < K),
                    other=0.0)

        # B tile [BLOCK_K, BLOCK_N]
        b = tl.load(b_ptrs,
                    mask=(k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)

        # acc += a @ b   ([BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N])
        acc += tl.dot(a, b)

        a_ptr_tile += BLOCK_K  # advance along k
        b_ptrs += BLOCK_K      # advance along k (weight already indexed by k)

    # ---- Store output [M, N] → out_ptr[ n * M + m ] --------------------------------
    # Mapping: out[0, n, i, j]  =  acc[ m=i*16+j, n ]
    # out_ptr index:  n * M + m
    out_ptrs = c_ptr + offs_n[None, :] * M + offs_m[:, None]

    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv_unfold_reshape(in_0, in_1):
    """
    Fused replacement for:
        conv2d(in_1, in_0, ...) → unfold(k=2,s=2) → reshape(1, 128, 4, -1)

    in_0 : [128, 256, 1, 1]  weight
    in_1 : [1, 256, 32, 32]  input
    Returns [1, 128, 4, 4]
    """
    M   = 8192   # 16 * 16 patches from unfold
    N   = 128    # output channels
    K   = 256    # input channels

    out = torch.empty((1, N, 16, 16), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    fused_conv_unfold_reshape_kernel[grid](
        in_1, in_0, out,
        M=M, N=N, K=K,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_conv_unfold_reshape