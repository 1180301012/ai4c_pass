import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_2 = torch.nn.functional.softmax(matmul, dim=-1, dtype=torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_2, p=0.0, training=False)
    to = tmp_4.to(torch.bfloat16)
    matmul_1 = torch.matmul(to, in_2)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['N', 'D_actual'],
)
@triton.jit
def flash_attn_bf16_kernel(
    Q, KT, V, Out,
    sq_b, sq_h, sq_n, sq_d,
    skt_b, skt_h, skt_d, skt_n,
    sv_b, sv_h, sv_n, sv_d,
    so_b, so_n, so_h, so_d,
    H, N, D_actual,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    bm = tl.program_id(1)
    b = bh // H
    h = bh % H

    m_start = bm * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block [BLOCK_M, BLOCK_D] in float32
    Q_base = Q + b * sq_b + h * sq_h
    q = tl.load(
        Q_base + offs_m[:, None] * sq_n + offs_d[None, :] * sq_d,
        mask=(offs_m[:, None] < N) & (offs_d[None, :] < D_actual),
        other=0.0
    ).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], -1e38, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    KT_base = KT + b * skt_b + h * skt_h
    V_base = V + b * sv_b + h * sv_h

    n_blocks = (N + BLOCK_N - 1) // BLOCK_N

    for j in range(n_blocks):
        start_n = j * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Load KT block [BLOCK_D, BLOCK_N] in float32
        kt = tl.load(
            KT_base + offs_d[:, None] * skt_d + offs_n[None, :] * skt_n,
            mask=(offs_d[:, None] < D_actual) & (n_mask[None, :]),
            other=0.0
        ).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N] (scale=1.0 applied to q)
        s = tl.dot(q, kt)
        s = tl.where(n_mask[None, :], s, -1e38)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(n_mask[None, :], p, tl.zeros_like(p))

        # Load V block [BLOCK_N, BLOCK_D] in float32
        v = tl.load(
            V_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=n_mask[:, None] & (offs_d[None, :] < D_actual),
            other=0.0
        ).to(tl.float32)

        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Normalize
    l_i = tl.where(l_i == 0, tl.ones_like(l_i), l_i)
    acc = acc / l_i[:, None]

    # Store output: layout [B, N, H, D] -> [1, 257, 16, 80] = [1, 257, 1280]
    Out_base = Out + b * so_b + h * so_h
    tl.store(
        Out_base + offs_m[:, None] * so_n + offs_d[None, :] * so_d,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < N) & (offs_d[None, :] < D_actual)
    )


@torch.fx.wrap
def flash_attention_bf16(in_0, in_1, in_2):
    """
    in_0: Q  [B, H, N, D]
    in_1: KT [B, H, D, N]  (K already transposed)
    in_2: V  [B, H, N, D]
    returns: [B, N, H*D]
    """
    B, H, N, D = in_0.shape
    BLOCK_D = 128  # next power of 2 >= D=80

    # Output: [B, N, H*D] = [1, 257, 1280]
    out = torch.empty(B, N, H * D, dtype=torch.bfloat16, device=in_0.device)

    sq_b, sq_h, sq_n, sq_d = in_0.stride()
    skt_b, skt_h, skt_d, skt_n = in_1.stride()
    sv_b, sv_h, sv_n, sv_d = in_2.stride()

    # Output strides for [B, N, H, D] layout
    so_b = N * H * D
    so_n = H * D
    so_h = D
    so_d = 1

    grid = lambda meta: (B * H, triton.cdiv(N, meta['BLOCK_M']))

    flash_attn_bf16_kernel[grid](
        in_0, in_1, in_2, out,
        sq_b, sq_h, sq_n, sq_d,
        skt_b, skt_h, skt_d, skt_n,
        sv_b, sv_h, sv_n, sv_d,
        so_b, so_n, so_h, so_d,
        H, N, D,
        BLOCK_D=BLOCK_D,
    )

    return out


def replacement_func():
    return flash_attention_bf16