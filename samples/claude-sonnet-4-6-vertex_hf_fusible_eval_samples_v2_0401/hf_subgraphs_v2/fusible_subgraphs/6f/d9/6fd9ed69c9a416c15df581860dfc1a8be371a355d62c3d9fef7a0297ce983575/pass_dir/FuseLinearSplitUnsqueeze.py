"""
Pass: FuseLinearSplitUnsqueeze
Fuses: linear(in_5, in_1, in_0) + [:, :256].view(-1,256) + [:, -256:].view(-1,256) + .unsqueeze(-2)
Pattern returns: (tmp_8 [M, 256], tmp_13 [M, 1, 256])
  - tmp_8  = second half of linear output, viewed
  - tmp_13 = first  half of linear output, viewed, unsqueezed
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _linear_split_kernel(
    x_ptr, w_ptr, b_ptr,
    out_second_ptr, out_first_ptr,   # [M, N//2] each
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes Y = X @ W^T + b  (X:[M,K], W:[N,K], b:[N])
    and writes:
      - Y[:, :N//2]    -> out_first  (contiguous [M, N//2])
      - Y[:, N//2:]    -> out_second (contiguous [M, N//2])
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    half_N = N // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # global column index in [0, N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for X tile and W tile (W stored as [N, K], so row=n, col=k)
    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    b_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        k_curr = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_curr[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_curr[:, None] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    m_mask = offs_m[:, None] < M

    # Write first half (cols 0..half_N-1) → out_first
    first_col_mask = (offs_n[None, :] < half_N) & (offs_n[None, :] < N)
    first_mask = m_mask & first_col_mask
    out_first_offs = offs_m[:, None] * half_N + offs_n[None, :]
    tl.store(out_first_ptr + out_first_offs, acc, mask=first_mask)

    # Write second half (cols half_N..N-1) → out_second
    second_col_mask = (offs_n[None, :] >= half_N) & (offs_n[None, :] < N)
    second_mask = m_mask & second_col_mask
    out_second_offs = offs_m[:, None] * half_N + (offs_n[None, :] - half_N)
    tl.store(out_second_ptr + out_second_offs, acc, mask=second_mask)


@torch.fx.wrap
def _run_linear_split_kernel(in_5, in_1, in_0):
    """
    Inner wrapped kernel launcher – becomes a single leaf node in FX graph.
    in_5: [M, K]  in_1: [N, K]  in_0: [N]
    Returns Python tuple: (out_second [M, N//2], out_first [M, N//2])
      out_second = linear output second half  (cols N//2 … N-1)
      out_first  = linear output first half   (cols 0   … N//2-1)
    """
    x = in_5
    w = in_1
    b = in_0

    M, K = x.shape[0], x.shape[1]
    N = w.shape[0]
    half_N = N // 2

    out_second = torch.empty((M, half_N), dtype=x.dtype, device=x.device)
    out_first  = torch.empty((M, half_N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _linear_split_kernel[grid](
        x, w, b,
        out_second, out_first,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )

    return (out_second, out_first)


# NOT @torch.fx.wrap – FX will trace this function so that result[0] and
# result[1].unsqueeze(-2) become *separate* nodes (= 2 copied_returning_nodes),
# matching the 2 returning_nodes from the pattern (tmp_8, tmp_13).
def fused_linear_split_unsqueeze(in_5, in_1, in_0):
    result         = _run_linear_split_kernel(in_5, in_1, in_0)
    out_second     = result[0]               # [M, N//2]  → tmp_8
    out_first_unsq = result[1].unsqueeze(-2) # [M, 1, N//2] → tmp_13
    return (out_second, out_first_unsq)


# ─── Pattern & replacement ────────────────────────────────────────────────────

def pattern(in_5, in_1, in_0):
    tmp_4  = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5  = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6  = tmp_5.view(-1, 256)
    tmp_7  = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8  = tmp_7.view(-1, 256)
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_8, tmp_13)


def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)


def replacement_func():
    return fused_linear_split_unsqueeze