"""
Pass: FuseLinearIndepBroadcastMul
Matches: linear(in_3, in_0) + in_2 * in_1  (two independent ops, both returned)
Used by: RTMPose graphs.
  in_0: weight [256, 512]
  in_1: scale  [256]
  in_2: input  [B, 17, 256]
  in_3: input  [B, 17, 512]
Returns: (in_2 * in_1, linear)
Strategy:
  - Use a Triton matmul kernel for the linear projection.
  - Use a Triton elementwise-broadcast kernel for the scale multiply.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: matrix multiply  C[m,n] = sum_k A[m,k] * B[n,k]  (B transposed)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Large-M configs (good for B=512, M=8704)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Medium-M configs (good for B=256, M=4352)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        # Small-M configs (good for B=1,8,64, M=17 to 1088)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[m,n] = sum_k A[m,k] * B[n,k]  (B is [N,K], i.e., weight matrix)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Pattern / replacement
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """
    Matches:
      linear = F.linear(in_3, in_0, None)   # matrix projection
      tmp_3  = in_2 * in_1                  # broadcast scale (independent)
      return (tmp_3, linear)
    """
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3  = in_2 * in_1
    return (tmp_3, linear)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def _triton_linear_proj(y, weight):
    """Opaque Triton matmul: computes F.linear(y, weight) = y @ weight.T"""
    N_out, K = weight.shape
    orig_shape = y.shape[:-1]
    M_y = y.numel() // K
    y_2d = y.reshape(M_y, K)
    lin_2d = torch.empty((M_y, N_out), dtype=y.dtype, device=y.device)
    grid = lambda meta: (
        triton.cdiv(M_y, meta['BLOCK_M']) * triton.cdiv(N_out, meta['BLOCK_N']),
    )
    _matmul_kernel[grid](
        y_2d, weight, lin_2d,
        M_y, N_out, K,
        y_2d.stride(0),   y_2d.stride(1),
        weight.stride(0), weight.stride(1),
        lin_2d.stride(0), lin_2d.stride(1),
    )
    return lin_2d.reshape(*orig_shape, N_out)


def _linear_and_scale(in_0, in_1, in_2, in_3):
    """
    NOT @torch.fx.wrap — FX traces into this function so the framework
    sees TWO separate output nodes matching the 2-output pattern.

    in_0: weight [N_out, K]
    in_1: scale  [N_out]
    in_2: x      [*, N_out]
    in_3: y      [*, K]
    Returns: (x * scale,  F.linear(y, weight))
    """
    linear = _triton_linear_proj(in_3, in_0)  # opaque Triton node
    scaled = in_2 * in_1                       # separate FX node
    return scaled, linear


def replacement_func():
    return _linear_and_scale