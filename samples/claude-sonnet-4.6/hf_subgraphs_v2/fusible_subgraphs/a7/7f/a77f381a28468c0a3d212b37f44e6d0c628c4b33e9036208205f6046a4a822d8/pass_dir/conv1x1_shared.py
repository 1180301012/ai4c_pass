"""
Shared Triton kernel for 1x1 convolution implemented as a GEMM.

Pattern: conv2d(in_1, in_0, None, stride, (0,0), (1,1), 1) → slice first N channels
Optimization: 1x1 conv is a GEMM; use Triton for high-performance matmul.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        # Small-K friendly configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
    feat_ptr, weight_ptr, out_ptr,
    M, N, K,
    HW_out, W_out,
    HW_in,  W_in,
    stride_s,          # spatial stride: 1 or 2
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes C[m, n] = sum_k A[m, k] * B[k, n]   (in float32 accum)

    A[m, k]:  feat[m_batch, k, h_out * stride_s, w_out * stride_s]
              = feat_ptr + m_batch * K * HW_in + k * HW_in + m_sp_in
    B[k, n]:  weight[n, k]
              = weight_ptr + n * K + k

    C → out[m_batch, n, h_out, w_out]
      = out_ptr + m_batch * N * HW_out + n * HW_out + m_sp_out
    """
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # Decompose m into batch index and output spatial index
    m_batch   = m_offs // HW_out        # (BLOCK_M,)
    m_sp_out  = m_offs  % HW_out        # (BLOCK_M,)
    h_out_idx = m_sp_out // W_out
    w_out_idx = m_sp_out  % W_out
    # Corresponding input spatial index (stride_s applied)
    m_sp_in   = h_out_idx * (stride_s * W_in) + w_out_idx * stride_s  # (BLOCK_M,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # (BLOCK_K,)
        k_mask = k_offs < K
        m_mask = m_offs < M
        n_mask = n_offs < N

        # ---- Load A tile (BLOCK_M, BLOCK_K) ----
        # A[m, k] = feat_ptr + m_batch * K * HW_in + k * HW_in + m_sp_in
        a_ptrs = (feat_ptr
                  + m_batch[:, None]  * (K * HW_in)
                  + k_offs[None, :]   * HW_in
                  + m_sp_in[:, None])
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # ---- Load B tile as (BLOCK_N, BLOCK_K) – coalesced in K dim ----
        # b_T[n, k] = weight[n, k] = weight_ptr + n * K + k  (stride-1 in k → coalesced)
        b_ptrs_T = weight_ptr + n_offs[:, None] * K + k_offs[None, :]
        b_T = tl.load(b_ptrs_T, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate in float32:  A (BLOCK_M, BLOCK_K) @ B^T^T = A @ B
        acc += tl.dot(a, tl.trans(b_T), out_dtype=tl.float32)

    # ---- Store C tile (BLOCK_M, BLOCK_N) ----
    out_ptrs = (out_ptr
                + m_batch[:, None]  * (N * HW_out)
                + n_offs[None, :]   * HW_out
                + m_sp_out[:, None])
    out_mask = (m_offs < M)[:, None] & (n_offs < N)[None, :]

    if IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


def run_conv1x1_slice(in_0, in_1, stride_s, slice_n, return_sf):
    """
    in_0       : weight  (Cout, Cin, 1, 1)  – may be on CPU
    in_1       : feature (N, Cin, H_in, W_in)
    stride_s   : 1 or 2
    slice_n    : number of channels for the slice output
    return_sf  : True  → return (slice_out, full_out)
                 False → return (full_out,  slice_out)

    NOTE: NOT decorated with @torch.fx.wrap – called only from within
    a @torch.fx.wrap + @torch.compiler.disable wrapper, so Dynamo
    never traces into here.
    """
    # Move weight to same device/dtype as feature map (no-op if already correct)
    weight = in_0.to(in_1.device).to(in_1.dtype).contiguous()
    feat   = in_1.contiguous()

    N_batch, Cin, H_in, W_in = feat.shape
    Cout = weight.shape[0]

    # For 1x1 conv with no padding: H_out = (H_in - 1) // stride + 1
    H_out = (H_in - 1) // stride_s + 1
    W_out = (W_in - 1) // stride_s + 1
    HW_in  = H_in * W_in
    HW_out = H_out * W_out

    M = N_batch * HW_out

    out = torch.empty((N_batch, Cout, H_out, W_out), dtype=feat.dtype, device=feat.device)

    # Flatten weight to 2-D (Cout, Cin)
    w2d = weight.view(Cout, Cin)

    IS_FP16 = feat.dtype == torch.float16
    IS_BF16 = feat.dtype == torch.bfloat16

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Cout, meta['BLOCK_N']),)

    conv1x1_gemm_kernel[grid](
        feat, w2d, out,
        M, Cout, Cin,
        HW_out, W_out,
        HW_in,  W_in,
        stride_s,
        IS_FP16, IS_BF16,
    )

    slice_out = out[:, :slice_n, :, :]

    if return_sf:
        return slice_out, out
    else:
        return out, slice_out