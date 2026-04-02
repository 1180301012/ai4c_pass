"""
Pass: FuseLinearGate_in2
Matches: out = in_1 * F.linear(in_2, in_0, None)
Used by: Gemma graphs where in_2 is the input to linear and in_1 is the gate.
Strategy: Fused Triton matmul + elementwise multiply (gate in epilogue).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_gate_kernel_v2(
    A_ptr, B_ptr, G_ptr, O_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_gm, stride_gn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """O[m,n] = G[m,n] * sum_k(A[m,k] * B[n,k])   (B transposed weight)"""
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
    g_ptrs = G_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    g = tl.load(g_ptrs, mask=out_mask, other=0.0)
    result = acc * g.to(tl.float32)
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, result.to(g.dtype), mask=out_mask)


def pattern(in_0, in_1, in_2):
    """
    Matches: out = in_1 * F.linear(in_2, in_0, None)
    in_0 = weight, in_1 = gate, in_2 = input
    """
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _fused_linear_gate_in2(in_0, in_1, in_2):
    """Fused: out = in_1 * F.linear(in_2, in_0)
    in_0: weight [N, K], in_1: gate [*, N], in_2: input [*, K]
    """
    weight = in_0
    gate   = in_1
    x      = in_2

    N, K = weight.shape
    orig_shape = x.shape[:-1]
    M = x.numel() // K

    x_2d    = x.reshape(M, K)
    gate_2d = gate.reshape(M, N)
    out_2d  = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    _fused_linear_gate_kernel_v2[grid](
        x_2d, weight, gate_2d, out_2d,
        M, N, K,
        x_2d.stride(0),    x_2d.stride(1),
        weight.stride(0),  weight.stride(1),
        gate_2d.stride(0), gate_2d.stride(1),
        out_2d.stride(0),  out_2d.stride(1),
    )
    return (out_2d.reshape(*orig_shape, N),)


def replacement_func():
    return _fused_linear_gate_in2