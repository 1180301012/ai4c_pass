import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches: linear(in_2, in_1, in_0).transpose(-1, -2) * in_3
    in_0: bias [D]
    in_1: weight [D, K]
    in_2: input [B, M, K]
    in_3: scale [B, D, N]
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 128, 'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 64,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 32,  'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_D': 64,  'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 32,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_D': 64,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_D': 64,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_D': 16,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_D': 32,  'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_D': 32,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_D': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_D': 32,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 2}),
    ],
    key=['M', 'D', 'K'],
)
@triton.jit
def _fused_linear_transpose_mul_kernel(
    a_ptr,        # in_2: [B, M, K]
    b_ptr,        # in_1 (weight): [D, K]
    bias_ptr,     # in_0 (bias): [D]
    scale_ptr,    # in_3: [B, D, N]
    c_ptr,        # output: [B, D, N]
    B, M, K, D, N,
    stride_ab, stride_am, stride_ak,
    stride_bd, stride_bk,
    stride_sb, stride_sd, stride_sn,
    stride_cb, stride_cd, stride_cn,
    BLOCK_M: tl.constexpr,  # tile over M (768)
    BLOCK_D: tl.constexpr,  # tile over D (196)
    BLOCK_K: tl.constexpr,  # tile over K (196)
):
    """
    c[b, d, m] = (sum_k a[b, m, k] * b[d, k] + bias[d]) * scale[b, d, m]
    pid_m → M tiles, pid_d → D tiles, pid_b → batch
    acc[BLOCK_M, BLOCK_D] stored transposed as [BLOCK_D, BLOCK_M]
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # fp32 accumulator [BLOCK_M, BLOCK_D]
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    a_base = a_ptr + pid_b * stride_ab
    b_base = b_ptr
    s_base = scale_ptr + pid_b * stride_sb
    c_base = c_ptr + pid_b * stride_cb

    # ----- tiled matmul over K -----
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)

        # a[m, k] → [BLOCK_M, BLOCK_K]
        a_mask = (m_off[:, None] < M) & (k_off[None, :] < K)
        a = tl.load(
            a_base + m_off[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=a_mask, other=0.0,
        )

        # b[d, k] → [BLOCK_D, BLOCK_K]
        b_mask = (d_off[:, None] < D) & (k_off[None, :] < K)
        b_tile = tl.load(
            b_base + d_off[:, None] * stride_bd + k_off[None, :] * stride_bk,
            mask=b_mask, other=0.0,
        )

        # acc += a @ b^T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_D] → [BLOCK_M, BLOCK_D]
        acc = acc + tl.dot(a, tl.trans(b_tile))

    # ----- add bias -----
    bias = tl.load(bias_ptr + d_off, mask=d_off < D, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    # ----- load scale[b, d, m] and fuse multiply -----
    s_mask = (d_off[:, None] < D) & (m_off[None, :] < N)
    scale = tl.load(
        s_base + d_off[:, None] * stride_sd + m_off[None, :] * stride_sn,
        mask=s_mask, other=0.0,
    )  # [BLOCK_D, BLOCK_M]

    # acc_T[BLOCK_D, BLOCK_M] * scale[BLOCK_D, BLOCK_M]
    out = tl.trans(acc).to(tl.float32) * scale.to(tl.float32)

    # ----- store: c[b, d, m], strides stride_cd=N, stride_cn=1 -----
    c_ptrs = c_base + d_off[:, None] * stride_cd + m_off[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=s_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0: bias   [D=196]
    in_1: weight [D=196, K=196]
    in_2: input  [B, M=768, K=196]
    in_3: scale  [B, D=196, N=768]
    out:          [B, D=196, N=768]
    """
    B, M, K = in_2.shape
    D = in_1.shape[0]
    N = M   # in_3 has shape [B, D, M], so N = M = in_2.shape[2] = 768

    out = torch.empty(B, D, N, dtype=in_3.dtype, device=in_2.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(D, meta['BLOCK_D']),
        B,
    )

    _fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, K, D, N,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul