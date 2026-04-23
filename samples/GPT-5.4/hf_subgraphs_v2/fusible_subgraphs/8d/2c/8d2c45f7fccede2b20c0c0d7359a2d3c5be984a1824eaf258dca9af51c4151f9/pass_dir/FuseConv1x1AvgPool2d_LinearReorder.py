import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "K"],
)
@triton.jit
def _avgpool2x2_reorder_kernel(
    x_ptr,
    pooled_ptr,
    M,
    K,
    OH,
    OW,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_pm,
    stride_pk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    n_idx = offs_m // (OH * OW)
    hw_idx = offs_m % (OH * OW)
    oh_idx = hw_idx // OW
    ow_idx = hw_idx % OW

    h0 = oh_idx * 2
    w0 = ow_idx * 2

    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x00_ptrs = x_ptr + (
        n_idx[:, None] * stride_xn
        + offs_k[None, :] * stride_xc
        + h0[:, None] * stride_xh
        + w0[:, None] * stride_xw
    )
    x01_ptrs = x_ptr + (
        n_idx[:, None] * stride_xn
        + offs_k[None, :] * stride_xc
        + h0[:, None] * stride_xh
        + (w0 + 1)[:, None] * stride_xw
    )
    x10_ptrs = x_ptr + (
        n_idx[:, None] * stride_xn
        + offs_k[None, :] * stride_xc
        + (h0 + 1)[:, None] * stride_xh
        + w0[:, None] * stride_xw
    )
    x11_ptrs = x_ptr + (
        n_idx[:, None] * stride_xn
        + offs_k[None, :] * stride_xc
        + (h0 + 1)[:, None] * stride_xh
        + (w0 + 1)[:, None] * stride_xw
    )

    x00 = tl.load(x00_ptrs, mask=mask, other=0.0)
    x01 = tl.load(x01_ptrs, mask=mask, other=0.0)
    x10 = tl.load(x10_ptrs, mask=mask, other=0.0)
    x11 = tl.load(x11_ptrs, mask=mask, other=0.0)
    pooled = (x00 + x01 + x10 + x11) * 0.25

    pooled_ptrs = pooled_ptr + offs_m[:, None] * stride_pm + offs_k[None, :] * stride_pk
    tl.store(pooled_ptrs, pooled, mask=mask)


@triton.jit
def _transpose_weight_kernel(
    w_ptr,
    wt_ptr,
    O,
    K,
    stride_wo,
    stride_wc,
    stride_wtn,
    stride_wtk,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (offs_o[None, :] < O) & (offs_k[:, None] < K)
    w_ptrs = w_ptr + offs_o[None, :] * stride_wo + offs_k[:, None] * stride_wc
    vals = tl.load(w_ptrs, mask=mask, other=0.0)

    wt_ptrs = wt_ptr + offs_k[:, None] * stride_wtk + offs_o[None, :] * stride_wtn
    tl.store(wt_ptrs, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_to_nchw_kernel(
    a_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    OH,
    OW,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_yn,
    stride_yo,
    stride_yh,
    stride_yw,
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

    k_iter = 0
    while k_iter < K:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_iter + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_iter + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & ((k_iter + offs_k)[None, :] < K)
        b_mask = ((k_iter + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        k_iter += BLOCK_K

    n_idx = offs_m // (OH * OW)
    hw_idx = offs_m % (OH * OW)
    oh_idx = hw_idx // OW
    ow_idx = hw_idx % OW

    y_ptrs = y_ptr + (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yo
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


_WT_CACHE = {}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _direct_conv1x1_avgpool2d_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    OH,
    OW,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wc,
    stride_yn,
    stride_yo,
    stride_yh,
    stride_yw,
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

    n_idx = offs_m // (OH * OW)
    hw_idx = offs_m % (OH * OW)
    oh_idx = hw_idx // OW
    ow_idx = hw_idx % OW
    h0 = oh_idx * 2
    w0 = ow_idx * 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_mask = k_iter + offs_k < K
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]

        x00 = tl.load(
            x_ptr + n_idx[:, None] * stride_xn + (k_iter + offs_k)[None, :] * stride_xc + h0[:, None] * stride_xh + w0[:, None] * stride_xw,
            mask=x_mask,
            other=0.0,
        )
        x01 = tl.load(
            x_ptr + n_idx[:, None] * stride_xn + (k_iter + offs_k)[None, :] * stride_xc + h0[:, None] * stride_xh + (w0 + 1)[:, None] * stride_xw,
            mask=x_mask,
            other=0.0,
        )
        x10 = tl.load(
            x_ptr + n_idx[:, None] * stride_xn + (k_iter + offs_k)[None, :] * stride_xc + (h0 + 1)[:, None] * stride_xh + w0[:, None] * stride_xw,
            mask=x_mask,
            other=0.0,
        )
        x11 = tl.load(
            x_ptr + n_idx[:, None] * stride_xn + (k_iter + offs_k)[None, :] * stride_xc + (h0 + 1)[:, None] * stride_xh + (w0 + 1)[:, None] * stride_xw,
            mask=x_mask,
            other=0.0,
        )
        x_avg = (x00 + x01 + x10 + x11) * 0.25

        w_ptrs = w_ptr + offs_n[None, :] * stride_wo + (k_iter + offs_k)[:, None] * stride_wc
        w_mask = (offs_n[None, :] < N) & k_mask[:, None]
        wv = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_avg, wv)
        k_iter += BLOCK_K

    y_ptrs = y_ptr + n_idx[:, None] * stride_yn + offs_n[None, :] * stride_yo + oh_idx[:, None] * stride_yh + ow_idx[:, None] * stride_yw
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)



@torch.fx.wrap
def _conv1x1_avgpool2d_wrapper(w, x):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    ww = x.shape[3]
    oc = w.shape[0]
    oh = h // 2
    ow = ww // 2
    m = n * oh * ow
    out = torch.empty((n, oc, oh, ow), device=x.device, dtype=x.dtype)

    use_reordered_path = (x.dtype == torch.float32) and (m >= 4096)

    if use_reordered_path:
        pooled = torch.empty((m, c), device=x.device, dtype=x.dtype)
        cache_key = (w.data_ptr(), w.shape[0], w.shape[1], str(w.dtype), str(w.device))
        wt = _WT_CACHE.get(cache_key)
        if wt is None:
            wt = torch.empty((c, oc), device=w.device, dtype=w.dtype)
            grid_trans = (triton.cdiv(oc, 32), triton.cdiv(c, 32))
            _transpose_weight_kernel[grid_trans](
                w,
                wt,
                oc,
                c,
                w.stride(0),
                w.stride(1),
                wt.stride(1),
                wt.stride(0),
                BLOCK_O=32,
                BLOCK_K=32,
            )
            _WT_CACHE[cache_key] = wt

        grid_pool = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(c, META["BLOCK_K"]))
        _avgpool2x2_reorder_kernel[grid_pool](
            x,
            pooled,
            m,
            c,
            oh,
            ow,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            pooled.stride(0),
            pooled.stride(1),
        )

        grid_mm = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(oc, META["BLOCK_N"]),)
        _matmul_to_nchw_kernel[grid_mm](
            pooled,
            wt,
            out,
            m,
            oc,
            c,
            oh,
            ow,
            pooled.stride(0),
            pooled.stride(1),
            wt.stride(0),
            wt.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
        return out

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(oc, META["BLOCK_N"]),)
    _direct_conv1x1_avgpool2d_kernel[grid](
        x,
        w,
        out,
        m,
        oc,
        c,
        oh,
        ow,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return _conv1x1_avgpool2d_wrapper