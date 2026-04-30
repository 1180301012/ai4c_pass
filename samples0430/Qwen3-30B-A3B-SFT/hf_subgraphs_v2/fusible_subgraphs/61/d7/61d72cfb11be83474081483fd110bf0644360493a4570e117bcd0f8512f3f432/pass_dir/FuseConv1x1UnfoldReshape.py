import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Fused GEMM kernel
#
# Layout mapping (M=1024, N=128, K=256, PATCH_W=32):
#   GEMM:  C[m, n]  = sum_k  input[m, k] * weight[n, k]
#          input : [M=1024, K=256]   (in_1 viewed as [H*W, Cin])
#          weight: [N=128,  K=256]   (in_0 viewed as [Cout, Cin])
#
#   output written at:
#     out[0, n, k_off, patch*32 + pos]
#   where m = patch*32 + pos,  k_off = m & 3
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=[],
)
@triton.jit
def fused_conv1x1_unfold_reshape_kernel(
    input_ptr,   # [1, K, H, W]  – treated as [M=H*W, K]
    weight_ptr,  # [N, K, 1, 1]  – treated as [N, K]
    output_ptr,  # [1, N, 4, M]
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # Load A tile: input[m, k]  →  ptr + m*1 + k*W (W=32*32=1024)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(
            input_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=a_mask, other=0.0
        )

        # Load B tile: weight[n, k]  →  ptr + n*K + k*1
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=b_mask, other=0.0
        )

        # acc += A[BLOCK_M, BLOCK_K] @ B.T[BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    # ── Write fused output ──────────────────────────────────────────────────
    # m → (patch, pos):  patch = m // 32,  pos = m % 32
    # out[0, n, k_off, patch*32 + pos]  where k_off = m & 3
    m_2d    = offs_m[:, None]                            # [BLOCK_M, 1]
    n_2d    = offs_n[None, :]                            # [1, BLOCK_N]
    patch   = m_2d // 32                                 # [BLOCK_M, 1]
    pos     = m_2d % 32                                  # [BLOCK_M, 1]
    k_off   = m_2d & 3                                   # [BLOCK_M, 1]

    out_offs = n_2d * (4 * M) + k_off * M + patch * 32 + pos   # [BLOCK_M, BLOCK_N]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(output_ptr + out_offs, acc.to(OUT_DTYPE), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Replacement wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv1x1_unfold_reshape(in_0, in_1):
    """
    in_0 : weight [Cout=128, Cin=256, 1, 1]
    in_1 : input  [1, Cin=256, H=32, W=32]
    returns       [1, Cout=128, 4, H*W=1024]
    """
    Cout = 128
    Cin  = 256
    H    = 32
    W    = 32
    M    = H * W   # 1024

    output = torch.empty((1, Cout, 4, M), dtype=in_1.dtype, device=in_1.device)

    # Map torch dtype → Triton dtype constant
    if in_1.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif in_1.dtype == torch.float16:
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    grid = lambda meta: (
        triton.cdiv(M,    meta['BLOCK_M']),
        triton.cdiv(Cout, meta['BLOCK_N']),
    )

    fused_conv1x1_unfold_reshape_kernel[grid](
        in_1,          # input_ptr
        in_0,          # weight_ptr
        output,        # output_ptr
        M, Cout, Cin,
        OUT_DTYPE=out_dtype,
    )

    return output


def replacement_func():
    return fused_conv1x1_unfold_reshape