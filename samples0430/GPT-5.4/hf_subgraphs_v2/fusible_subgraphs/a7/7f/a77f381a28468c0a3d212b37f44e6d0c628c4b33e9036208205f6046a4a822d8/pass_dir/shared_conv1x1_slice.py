import math
import torch
import triton
import triton.language as tl


# Shared route constants used by all pass files.
ROUTE_CONV1X1_RET_SLICE_FULL = "conv1x1_ret_slice_full"
ROUTE_CONV1X1_RET_FULL_SLICE = "conv1x1_ret_full_slice"


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_nhwc_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk
        w_ptrs = w_ptr + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    out = acc.to(tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_nhwc_gemm_cast_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk
        w_ptrs = w_ptr + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    out = acc.to(OUT_DTYPE)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _select_kernel(x_dtype, w_dtype, out_dtype):
    if out_dtype == torch.float32:
        return _conv1x1_nhwc_gemm_kernel, None
    if out_dtype == torch.float16:
        return _conv1x1_nhwc_gemm_cast_kernel, tl.float16
    if out_dtype == torch.bfloat16:
        return _conv1x1_nhwc_gemm_cast_kernel, tl.bfloat16
    return _conv1x1_nhwc_gemm_kernel, None


def _launch_conv1x1_flat(x_flat, w_flat, out_flat):
    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_flat.shape[1]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    kernel, out_dtype = _select_kernel(x_flat.dtype, w_flat.dtype, out_flat.dtype)
    if out_dtype is None:
        kernel[grid](
            x_flat,
            w_flat,
            out_flat,
            M,
            N,
            K,
            x_flat.stride(0),
            x_flat.stride(1),
            w_flat.stride(0),
            w_flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
        )
    else:
        kernel[grid](
            x_flat,
            w_flat,
            out_flat,
            M,
            N,
            K,
            x_flat.stride(0),
            x_flat.stride(1),
            w_flat.stride(0),
            w_flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            OUT_DTYPE=out_dtype,
        )


@torch.fx.wrap
def shared_conv1x1_slice_dispatch(in_0, in_1, slice_channels, stride_hw, return_order):
    # PoisonDispatch warmup path: only factory ops are allowed there.
    if type(in_0) is not torch.Tensor or type(in_1) is not torch.Tensor:
        n = in_1.shape[0]
        cin = in_1.shape[1]
        h = in_1.shape[2]
        w = in_1.shape[3]
        cout = in_0.shape[0]
        sh = stride_hw[0]
        sw = stride_hw[1]
        oh = (h - 1) // sh + 1
        ow = (w - 1) // sw + 1
        full = torch.empty((n, cout, oh, ow), device=in_1.device, dtype=in_1.dtype)
        sliced = torch.empty((n, slice_channels, oh, ow), device=in_1.device, dtype=in_1.dtype)
        if return_order == 0:
            return sliced, full
        return full, sliced

    n, cin, h, w = in_1.shape
    cout = in_0.shape[0]
    sh = stride_hw[0]
    sw = stride_hw[1]
    oh = (h - 1) // sh + 1
    ow = (w - 1) // sw + 1

    x_nhwc = in_1.permute(0, 2, 3, 1)
    if sh != 1 or sw != 1:
        x_nhwc = x_nhwc[:, ::sh, ::sw, :]
    x_flat = x_nhwc.contiguous().view(n * oh * ow, cin)
    w_flat = in_0[:, :, 0, 0].contiguous().transpose(0, 1).contiguous()

    full_flat = torch.empty((n * oh * ow, cout), device=in_1.device, dtype=in_1.dtype)
    _launch_conv1x1_flat(x_flat, w_flat, full_flat)

    full_nhwc = full_flat.view(n, oh, ow, cout)
    full = full_nhwc.permute(0, 3, 1, 2)
    sliced = full[:, :slice_channels, :, :]

    if return_order == 0:
        return sliced, full
    return full, sliced