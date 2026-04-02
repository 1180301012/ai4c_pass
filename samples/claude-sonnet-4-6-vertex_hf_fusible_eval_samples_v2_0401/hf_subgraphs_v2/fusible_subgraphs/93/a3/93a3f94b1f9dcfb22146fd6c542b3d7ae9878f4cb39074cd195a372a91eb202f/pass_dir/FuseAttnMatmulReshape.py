import torch
import triton
import triton.language as tl


def pattern(attn_weights, v):
    """
    Matches the second matmul + transpose + reshape tail.
    Works for all dtypes (bf16, fp16, fp32) since the operations
    are dtype-agnostic.
    """
    result = torch.matmul(attn_weights, v)
    transposed = result.transpose(1, 2)
    cont1 = transposed.contiguous()
    reshaped = cont1.reshape(1, 257, -1)
    cont2 = reshaped.contiguous()
    return (cont2,)


def replacement_args(attn_weights, v):
    return (attn_weights, v)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    ],
    key=['N', 'D_actual', 'S'],
)
@triton.jit
def matmul_transpose_reshape_kernel(
    Attn_ptr, V_ptr, Out_ptr,
    # Attn strides [B, H, N, S]
    sa_b, sa_h, sa_n, sa_s,
    # V strides [B, H, S, D]
    sv_b, sv_h, sv_s, sv_d,
    # Out strides: written as [B, N, H, D] = [B, N, H*D]
    so_b, so_n, so_h, so_d,
    B, H, N, S, D_actual,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute: Out[b, n, h, d] = sum_s Attn[b, h, n, s] * V[b, h, s, d]
    Output written in [B, N, H, D] layout (transposed from [B, H, N, D]).
    Uses native dtype (bf16/fp16) for tensor-core acceleration via tl.dot.
    Large BLOCK_M (e.g. 256) minimizes V re-reads: V is read only
    ceil(N/BLOCK_M) times instead of (N/BLOCK_M) * ceil(S/BLOCK_N) times.
    """
    bh = tl.program_id(0)
    bm = tl.program_id(1)
    b = bh // H
    h = bh % H

    m_start = bm * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Accumulator for output [BLOCK_M, BLOCK_D] in float32
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    Attn_base = Attn_ptr + b * sa_b + h * sa_h
    V_base = V_ptr + b * sv_b + h * sv_h

    n_blocks = (S + BLOCK_N - 1) // BLOCK_N
    for j in range(n_blocks):
        start_s = j * BLOCK_N
        offs_s = start_s + tl.arange(0, BLOCK_N)
        s_mask = offs_s < S

        # Load Attn block [BLOCK_M, BLOCK_N] - native dtype for tensor cores
        attn = tl.load(
            Attn_base + offs_m[:, None] * sa_n + offs_s[None, :] * sa_s,
            mask=(offs_m[:, None] < N) & (s_mask[None, :]),
            other=0.0
        )

        # Load V block [BLOCK_N, BLOCK_D] - native dtype for tensor cores
        v = tl.load(
            V_base + offs_s[:, None] * sv_s + offs_d[None, :] * sv_d,
            mask=s_mask[:, None] & (offs_d[None, :] < D_actual),
            other=0.0
        )

        # tl.dot uses tensor cores for bf16/fp16 and accumulates in float32
        acc += tl.dot(attn, v, out_dtype=tl.float32)

    # Store output in [B, N, H, D] layout
    Out_base = Out_ptr + b * so_b + h * so_h
    tl.store(
        Out_base + offs_m[:, None] * so_n + offs_d[None, :] * so_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < N) & (offs_d[None, :] < D_actual)
    )


@torch.fx.wrap
def fused_matmul_transpose_reshape(attn_weights, v):
    """
    attn_weights: [B, H, N, S]  (e.g. [1, 16, 257, 257])
    v:            [B, H, S, D]  (e.g. [1, 16, 257, 80])
    returns:      [B, N, H*D]   (e.g. [1, 257, 1280])
    Uses Triton kernel that writes directly in [B, N, H*D] layout,
    avoiding the expensive transpose + contiguous copy step.
    With BLOCK_M=256, V is read only ceil(N/256) = 2 times per head,
    making total memory bandwidth nearly equal to cuBLAS + copy approach.
    """
    B, H, N, S = attn_weights.shape
    D = v.shape[-1]
    BLOCK_D = 128  # next power of 2 >= D=80

    out = torch.empty(B, N, H * D, dtype=attn_weights.dtype, device=attn_weights.device)

    sa_b, sa_h, sa_n, sa_s = attn_weights.stride()
    sv_b, sv_h, sv_s, sv_d = v.stride()

    # Output strides for [B, N, H, D] layout
    so_b = N * H * D
    so_n = H * D
    so_h = D
    so_d = 1

    grid = lambda meta: (B * H, triton.cdiv(N, meta['BLOCK_M']))

    matmul_transpose_reshape_kernel[grid](
        attn_weights, v, out,
        sa_b, sa_h, sa_n, sa_s,
        sv_b, sv_h, sv_s, sv_d,
        so_b, so_n, so_h, so_d,
        B, H, N, S, D,
        BLOCK_D=BLOCK_D,
    )

    return out


def replacement_func():
    return fused_matmul_transpose_reshape