"""
Pass: FuseReshapeLinearSplit
Fuses: in_4.reshape(300,-1,256) + linear(tmp_9, in_3, in_2) + [...,:256] + [...,-256:]
Pattern returns: (tmp_11 [300,1,256], tmp_12 [300,1,256])
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
def _reshape_linear_split_kernel(
    x_ptr, w_ptr, b_ptr,
    out_first_ptr, out_second_ptr,   # [M, N//2] each (will be reshaped to 3D)
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
      - Y[:, :N//2]  -> out_first  (contiguous [M, N//2])
      - Y[:, N//2:]  -> out_second (contiguous [M, N//2])
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    half_N = N // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # global column index in [0, N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for X tile and W tile (W stored as [N, K])
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
def _run_reshape_linear_kernel(in_4, in_3, in_2):
    """
    Inner wrapped kernel launcher – becomes a single leaf node in FX graph.
    in_4: [1, 150, 1, 512]  →  internally reshaped to [300, 256] for the GEMM
    in_3: [512, 256]  weight
    in_2: [512]       bias
    Returns Python tuple: (out_first [300, 256], out_second [300, 256])
      out_first  = linear output first  half  (cols 0   … N//2-1)
      out_second = linear output second half  (cols N//2 … N-1)
    """
    # Flatten: [1,150,1,512] → reshape(300,-1,256) → [300,1,256] → flatten to [300,256]
    x = in_4.reshape(300, -1, 256).reshape(-1, 256)  # [300, 256]
    w = in_3    # [512, 256]
    b = in_2    # [512]

    M  = x.shape[0]   # 300
    K  = x.shape[1]   # 256
    N  = w.shape[0]   # 512
    half_N = N // 2   # 256

    out_first  = torch.empty((M, half_N), dtype=x.dtype, device=x.device)
    out_second = torch.empty((M, half_N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _reshape_linear_split_kernel[grid](
        x, w, b,
        out_first, out_second,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )

    return (out_first, out_second)


# NOT @torch.fx.wrap – FX traces this so result[0].reshape and result[1].reshape
# become *separate* nodes (= 2 copied_returning_nodes), matching the 2
# returning_nodes from the pattern (tmp_11, tmp_12).
def fused_reshape_linear_split(in_4, in_3, in_2):
    result = _run_reshape_linear_kernel(in_4, in_3, in_2)
    # Restore the mid-dimension that the original linear preserved: [300,256] → [300,1,256]
    tmp_11 = result[0].reshape(300, -1, 256)  # first  half → tmp_11
    tmp_12 = result[1].reshape(300, -1, 256)  # second half → tmp_12
    return (tmp_11, tmp_12)


# ─── Pattern & replacement ────────────────────────────────────────────────────

def pattern(in_4, in_3, in_2):
    tmp_9  = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    return (tmp_11, tmp_12)


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)


def replacement_func():
    return fused_reshape_linear_split