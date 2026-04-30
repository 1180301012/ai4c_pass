import torch
import triton
import triton.language as tl


@triton.jit
def _row_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    rows,
    cols,
    stride_xr,
    stride_xc,
    stride_or,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xr + offs_n[None, :] * stride_xc
    s_ptrs = scale_ptr + offs_n
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    s = tl.load(s_ptrs, mask=offs_n < cols, other=0.0)
    out = x * s[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_or + offs_n[None, :] * stride_oc
    tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_mul_kernel(
    a_ptr,
    w_ptr,
    g_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
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

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0.0)
        acc += tl.dot(a, w)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    gate = tl.load(g_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc * gate, mask=mask)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "row_scale":
        x, scale, _ = args
        out = torch.empty_like(x)
        cols = scale.shape[0]
        rows = x.numel() // cols
        grid = lambda META: (triton.cdiv(rows, META["BLOCK_M"]), triton.cdiv(cols, META["BLOCK_N"]))
        _row_scale_kernel[grid](
            x,
            scale,
            out,
            rows,
            cols,
            cols,
            1,
            cols,
            1,
            BLOCK_M=64,
            BLOCK_N=256,
        )
        return out
    elif route == "linear_mul":
        w, x, y, _ = args
        K = w.shape[1]
        N = w.shape[0]
        M = x.numel() // K
        out = torch.empty_like(y)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _fused_linear_mul_kernel[grid](
            x,
            w,
            y,
            out,
            M,
            N,
            K,
            K,
            1,
            K,
            1,
            N,
            1,
            N,
            1,
        )
        return out
    else:
        raise RuntimeError(f"Unknown route: {route}")