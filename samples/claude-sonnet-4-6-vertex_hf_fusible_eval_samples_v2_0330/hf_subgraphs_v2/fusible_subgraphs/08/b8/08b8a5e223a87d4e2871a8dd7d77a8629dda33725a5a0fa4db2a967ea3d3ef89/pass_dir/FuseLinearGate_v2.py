"""
Pattern B2: linear(in_1, in_0, None) followed by in_2 * linear_result
Matches the SmolLM3-3B graph pattern.
Fuses GEMM + element-wise multiply into a single Triton kernel.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.ops.aten.linear.default(in_1, in_0, None)
    tmp_2 = torch.ops.aten.mul.Tensor(in_2, linear)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def smollm_fused_gemm_mul_kernel(
    A_ptr, B_ptr, C_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fused: Out = (A @ B^T) * C"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m % M)[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + (offs_n % N)[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=mask, other=0.0)

    out = acc.to(c.dtype) * c
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def smollm_fused_linear_gate(in_0, in_1, in_2):
    """
    Fused: result = in_2 * linear(in_1, in_0)
    in_0: weight [N, K]
    in_1: input  [*, M, K]
    in_2: gate   [*, M, N]
    """
    in_shape = in_1.shape
    M = in_1.numel() // in_1.shape[-1]
    K = in_1.shape[-1]
    N = in_0.shape[0]

    A = in_1.reshape(M, K).contiguous()
    B = in_0.contiguous()
    C = in_2.reshape(M, N).contiguous()
    Out = torch.empty(M, N, dtype=in_1.dtype, device=in_1.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    smollm_fused_gemm_mul_kernel[grid](
        A, B, C, Out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        Out.stride(0), Out.stride(1),
    )

    out_shape = in_shape[:-1] + (N,)
    return Out.reshape(out_shape)


def replacement_func():
    return smollm_fused_linear_gate