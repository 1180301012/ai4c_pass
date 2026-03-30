"""
Pattern A: Independent linear(in_3, in_0, None) + in_2 * in_1
Matches the rtmpose-l graph pattern where linear and multiply are independent.
Uses a pure Triton GEMM kernel for the linear part and a Triton multiply kernel.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.ops.aten.linear.default(in_3, in_0, None)
    tmp_3 = torch.ops.aten.mul.Tensor(in_2, in_1)
    return tmp_3, linear


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ── GEMM-only kernel (Out = A @ B^T) ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_only_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Out = A @ B^T  with float32 accumulation, stored as OUTPUT_DTYPE."""
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
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(OUTPUT_DTYPE), mask=mask)


# ── broadcast-multiply kernel ──────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
    ],
    key=['n_elements', 'n_scale'],
)
@triton.jit
def mul_broadcast_1d_kernel(
    x_ptr, scale_ptr, out_ptr,
    n_elements, n_scale,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = x[i] * scale[i % n_scale]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets % n_scale, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


@torch.fx.wrap
def linear_and_mul(in_0, in_1, in_2, in_3):
    """
    in_0: weight [N, K]   in_1: scale [N]
    in_2: x      [*, S, N]  in_3: x_in [*, S, K]
    Returns: (in_2 * in_1,  in_3 @ in_0.T)
    """
    # Select Triton output dtype from the tensor dtype
    if in_3.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif in_3.dtype == torch.float16:
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    # ---- GEMM: linear_out = in_3 @ in_0.T ----
    in3_shape = in_3.shape
    M_total = in_3.numel() // in_3.shape[-1]
    K_in    = in_3.shape[-1]
    N_out   = in_0.shape[0]

    A = in_3.reshape(M_total, K_in).contiguous()
    B = in_0.contiguous()
    linear_2d = torch.empty(M_total, N_out, dtype=in_3.dtype, device=in_3.device)

    grid_g = lambda META: (triton.cdiv(M_total, META['BLOCK_M']) * triton.cdiv(N_out, META['BLOCK_N']),)
    gemm_only_kernel[grid_g](
        A, B, linear_2d,
        M_total, N_out, K_in,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        linear_2d.stride(0), linear_2d.stride(1),
        OUTPUT_DTYPE=out_dtype,
    )
    linear_out = linear_2d.reshape(in3_shape[:-1] + (N_out,))

    # ---- Multiply: mul_out = in_2 * in_1 (broadcast) ----
    in_2_c = in_2.contiguous()
    in_1_c = in_1.contiguous()
    mul_out = torch.empty_like(in_2_c)
    n_elem  = in_2_c.numel()
    n_scale = in_1_c.numel()

    grid_m = lambda META: (triton.cdiv(n_elem, META['BLOCK_SIZE']),)
    mul_broadcast_1d_kernel[grid_m](
        in_2_c, in_1_c, mul_out,
        n_elem, n_scale,
    )

    return mul_out, linear_out


def replacement_func():
    return linear_and_mul