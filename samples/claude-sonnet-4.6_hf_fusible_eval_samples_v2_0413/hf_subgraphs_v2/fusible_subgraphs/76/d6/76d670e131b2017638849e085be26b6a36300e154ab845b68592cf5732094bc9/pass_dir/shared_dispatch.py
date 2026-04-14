"""
Shared Triton kernels and dispatch function for BN-inference and linear GEMM passes.
"""
import torch
import triton
import triton.language as tl


# ============================================================
# Batch-Norm Inference kernel  (2-D grid: B x cdiv(C, BLOCK_C))
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 512}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C'],
)
@triton.jit
def _bn_kernel(
    x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b_idx  = tl.program_id(0)
    cb_idx = tl.program_id(1)
    c_offs = cb_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    mean  = tl.load(mean_ptr  + c_offs, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + c_offs, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_offs, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_offs, mask=mask, other=0.0).to(tl.float32)

    scale    = gamma * tl.rsqrt(var + 1e-5)
    bias_val = beta - mean * scale

    row_base = b_idx * C
    x   = tl.load(x_ptr + row_base + c_offs, mask=mask, other=0.0).to(tl.float32)
    out = x * scale + bias_val

    if IS_FP16:
        tl.store(out_ptr + row_base + c_offs, out.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_base + c_offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_base + c_offs, out, mask=mask)


def _fused_bn(input, running_mean, running_var, weight, bias):
    B, C = input.shape[0], input.shape[1]
    out  = torch.empty_like(input)
    is_fp16 = input.dtype == torch.float16
    is_bf16 = input.dtype == torch.bfloat16
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
    _bn_kernel[grid](
        input, running_mean, running_var, weight, bias, out,
        B, C, IS_FP16=is_fp16, IS_BF16=is_bf16,
    )
    return out


# ============================================================
# Linear GEMM: C[M,N] = A[M,K] @ W[N,K]^T + bias[N]
# Load W^T as [BLOCK_K, BLOCK_N] via strided access.
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel(
    a_ptr, w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id     = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs  = a_ptr + (offs_m[:, None] % M) * stride_am + offs_k[None, :] * stride_ak
    wt_ptrs = w_ptr + offs_k[:, None] * stride_wk + (offs_n[None, :] % N) * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a  = tl.load(a_ptrs,  mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        wt = tl.load(wt_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, wt)
        a_ptrs  += BLOCK_K * stride_ak
        wt_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + out_m[:, None] * stride_cm + out_n[None, :] * stride_cn
    c_mask = (out_m[:, None] < M) & (out_n[None, :] < N)

    if IS_FP16:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif IS_BF16:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


def _fused_linear(input, weight, bias):
    M, K = input.shape[0], input.shape[1]
    N    = weight.shape[0]
    out  = torch.empty((M, N), dtype=input.dtype, device=input.device)
    is_fp16 = input.dtype == torch.float16
    is_bf16 = input.dtype == torch.bfloat16
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    _linear_kernel[grid](
        input, weight, bias, out,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        IS_FP16=is_fp16, IS_BF16=is_bf16,
    )
    return out


# ============================================================
# Shared dispatch (single @torch.fx.wrap for all passes)
# BN:     dispatch(input, mean, var, gamma, beta, "bn")
# Linear: dispatch(input, weight, bias, None, None, "linear")
# ============================================================
@torch.fx.wrap
def dispatch(arg0, arg1, arg2, arg3=None, arg4=None, route="bn"):
    if route == "bn":
        return _fused_bn(arg0, arg1, arg2, arg3, arg4)
    else:
        return _fused_linear(arg0, arg1, arg2)
        # ---- Linear GEMM: C[M,N] = A[M,K] @ W[N,K]^T + bias[N] ----
        lin_pid = pid - P_BN

        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id      = lin_pid // num_pid_in_group
        first_pid_m   = group_id * GROUP_M
        group_size_m  = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (lin_pid % group_size_m)
        pid_n = (lin_pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # A[offs_m, offs_k]  coalesced along K
        a_ptrs = a_ptr + (offs_m[:, None] % M) * stride_am + offs_k[None, :] * stride_ak
        # W^T[offs_k, offs_n] = W[offs_n, offs_k]: load as [BLOCK_K, BLOCK_N]
        # stride_wk=1 (col stride), stride_wn=K (row stride)
        wt_ptrs = w_ptr + offs_k[:, None] * stride_wk + (offs_n[None, :] % N) * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
            w = tl.load(wt_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, w)
            a_ptrs  += BLOCK_K * stride_ak
            wt_ptrs += BLOCK_K * stride_wk

        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

        out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = out_lin_ptr + out_m[:, None] * stride_cm + out_n[None, :] * stride_cn
        c_mask = (out_m[:, None] < M) & (out_n[None, :] < N)

        if IS_FP16:
            tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
        elif IS_BF16:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
        else:
            tl.store(c_ptrs, acc, mask=c_mask)


def _combined_linear_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    in_0 = running_mean  [C]
    in_1 = running_var   [C]
    in_2 = bn_bias/beta  [C]
    in_3 = bn_weight/gamma [C]
    in_4 = linear_bias   [N]
    in_5 = linear_weight [N, K]
    in_6 = linear_input  [M, K]
    in_7 = bn_input      [B, C]
    """
    M, K = in_6.shape[0], in_6.shape[1]
    N    = in_5.shape[0]
    B, C = in_7.shape[0], in_7.shape[1]

    out_lin = torch.empty((M, N), dtype=in_6.dtype, device=in_6.device)
    out_bn  = torch.empty_like(in_7)

    is_fp16 = in_6.dtype == torch.float16
    is_bf16 = in_6.dtype == torch.bfloat16

    # --- Choose linear tile size based on M (no autotune, avoids Python overhead) ---
    BLOCK_C = 128
    if M <= 4:
        BM, BN, BK, NW, NS = 16, 64,  64, 4, 4
    elif M <= 16:
        BM, BN, BK, NW, NS = 16, 128, 64, 4, 4
    elif M <= 32:
        BM, BN, BK, NW, NS = 32, 128, 64, 4, 4
    elif M <= 64:
        BM, BN, BK, NW, NS = 64, 128, 64, 8, 4
    else:
        BM, BN, BK, NW, NS = 128, 128, 64, 8, 4

    P_BN  = B * triton.cdiv(C, BLOCK_C)
    P_LIN = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    grid  = (P_BN + P_LIN,)

    _combined_bn_linear_kernel[grid](
        in_7,  in_0, in_1, in_3, in_2, out_bn,
        B, C,
        in_6,  in_5, in_4, out_lin,
        M, N, K,
        in_6.stride(0),  in_6.stride(1),
        in_5.stride(0),  in_5.stride(1),
        out_lin.stride(0), out_lin.stride(1),
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        BLOCK_C=BLOCK_C, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
        num_warps=NW,
        num_stages=NS,
    )
    return (out_lin, out_bn)


# ============================================================
# Stand-alone BN (used as fallback for single-op passes)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 512}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C'],
)
@triton.jit
def _bn_kernel(
    x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b_idx  = tl.program_id(0)
    cb_idx = tl.program_id(1)
    c_offs = cb_idx * BLOCK_C + tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    mean  = tl.load(mean_ptr  + c_offs, mask=mask, other=0.0).to(tl.float32)
    var   = tl.load(var_ptr   + c_offs, mask=mask, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c_offs, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + c_offs, mask=mask, other=0.0).to(tl.float32)

    scale    = gamma * tl.rsqrt(var + 1e-5)
    bias_val = beta - mean * scale

    row_base = b_idx * C
    x   = tl.load(x_ptr + row_base + c_offs, mask=mask, other=0.0).to(tl.float32)
    out = x * scale + bias_val

    if IS_FP16:
        tl.store(out_ptr + row_base + c_offs, out.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_base + c_offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_base + c_offs, out, mask=mask)


def _fused_bn(input, running_mean, running_var, weight, bias):
    B, C = input.shape[0], input.shape[1]
    out  = torch.empty_like(input)
    is_fp16 = input.dtype == torch.float16
    is_bf16 = input.dtype == torch.bfloat16
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
    _bn_kernel[grid](
        input, running_mean, running_var, weight, bias, out,
        B, C,
        IS_FP16=is_fp16, IS_BF16=is_bf16,
    )
    return out


# ============================================================
# Stand-alone linear GEMM (used as fallback)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel(
    a_ptr, w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id     = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] % M) * stride_am + offs_k[None, :] * stride_ak
    # W^T[offs_k, offs_n] loaded as [BLOCK_K, BLOCK_N]
    wt_ptrs = w_ptr + offs_k[:, None] * stride_wk + (offs_n[None, :] % N) * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a  = tl.load(a_ptrs,  mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        wt = tl.load(wt_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, wt)
        a_ptrs  += BLOCK_K * stride_ak
        wt_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + out_m[:, None] * stride_cm + out_n[None, :] * stride_cn
    c_mask = (out_m[:, None] < M) & (out_n[None, :] < N)

    if IS_FP16:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif IS_BF16:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


def _fused_linear(input, weight, bias):
    M, K = input.shape[0], input.shape[1]
    N    = weight.shape[0]
    out  = torch.empty((M, N), dtype=input.dtype, device=input.device)
    is_fp16 = input.dtype == torch.float16
    is_bf16 = input.dtype == torch.bfloat16
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    _linear_kernel[grid](
        input, weight, bias, out,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        IS_FP16=is_fp16, IS_BF16=is_bf16,
    )
    return out


# ============================================================
# Shared dispatch (single @torch.fx.wrap, all routes here)
# BN route:       (in0, in1, in2, in3, in4, None, None, None, "bn")
# Linear route:   (in0, in1, in2, None, None, None, None, None, "linear")
# Combined route: (in0, in1, in2, in3, in4, in5, in6, in7, "combined")
# ============================================================
@torch.fx.wrap
def dispatch(arg0, arg1, arg2, arg3=None, arg4=None, route="bn"):
    if route == "bn":
        return _fused_bn(arg0, arg1, arg2, arg3, arg4)
    else:
        return _fused_linear(arg0, arg1, arg2)