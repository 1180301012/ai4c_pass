import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=5, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_mul_kernel(
    a_ptr,
    w_ptr,
    g_ptr,
    out_ptr,
    M,
    N,
    K,
    S,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_w0,
    stride_w1,
    stride_g0,
    stride_g1,
    stride_g2,
    stride_o0,
    stride_o1,
    stride_o2,
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
    rows_b = offs_m // S
    rows_s = offs_m % S

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows_b[:, None] * stride_a0 + rows_s[:, None] * stride_a1 + offs_k[None, :] * stride_a2
        w_ptrs = w_ptr + offs_n[None, :] * stride_w0 + offs_k[:, None] * stride_w1
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(a, w)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    g_ptrs = g_ptr + rows_b[:, None] * stride_g0 + rows_s[:, None] * stride_g1 + offs_n[None, :] * stride_g2
    g = tl.load(g_ptrs, mask=out_mask, other=0.0)
    out = acc * g
    out_ptrs = out_ptr + rows_b[:, None] * stride_o0 + rows_s[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, out, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=5, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_only_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    S,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    stride_o2,
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
    rows_b = offs_m // S
    rows_s = offs_m % S

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows_b[:, None] * stride_a0 + rows_s[:, None] * stride_a1 + offs_k[None, :] * stride_a2
        w_ptrs = w_ptr + offs_n[None, :] * stride_w0 + offs_k[:, None] * stride_w1
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(a, w)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + rows_b[:, None] * stride_o0 + rows_s[:, None] * stride_o1 + offs_n[None, :] * stride_o2
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def broadcast_scale_mul_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % channels
    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * s, mask=mask)


def _extract_3d_meta(x):
    dim = x.dim()
    if dim == 3:
        return (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
        )
    if dim == 2:
        return (
            1,
            x.shape[0],
            x.shape[1],
            0,
            x.stride(0),
            x.stride(1),
        )
    raise RuntimeError(f"Expected a 2D or 3D tensor, got dim={dim}")


def _fused_linear_mul(weight, hidden, gate):
    out = torch.empty_like(gate)

    _, hidden_s, hidden_k, stride_a0, stride_a1, stride_a2 = _extract_3d_meta(hidden)
    _, gate_s, gate_n, stride_g0, stride_g1, stride_g2 = _extract_3d_meta(gate)
    _, _, _, stride_o0, stride_o1, stride_o2 = _extract_3d_meta(out)

    if hidden_s != gate_s:
        raise RuntimeError("Hidden state and gate leading dimensions must match")
    if hidden_k != weight.shape[1]:
        raise RuntimeError("Linear K dimension mismatch")
    if gate_n != weight.shape[0]:
        raise RuntimeError("Linear N dimension mismatch")

    m = hidden.numel() // hidden_k
    n = weight.shape[0]
    k = hidden_k
    s = hidden_s

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
    fused_linear_mul_kernel[grid](
        hidden,
        weight,
        gate,
        out,
        m,
        n,
        k,
        s,
        stride_a0,
        stride_a1,
        stride_a2,
        weight.stride(0),
        weight.stride(1),
        stride_g0,
        stride_g1,
        stride_g2,
        stride_o0,
        stride_o1,
        stride_o2,
    )
    return out


def _linear_only(weight, hidden, out_like):
    out = torch.empty_like(out_like)

    _, hidden_s, hidden_k, stride_a0, stride_a1, stride_a2 = _extract_3d_meta(hidden)
    _, out_s, out_n, _, _, _ = _extract_3d_meta(out)
    _, _, _, stride_o0, stride_o1, stride_o2 = _extract_3d_meta(out)

    if hidden_s != out_s:
        raise RuntimeError("Hidden state and output leading dimensions must match")
    if hidden_k != weight.shape[1]:
        raise RuntimeError("Linear K dimension mismatch")
    if out_n != weight.shape[0]:
        raise RuntimeError("Linear N dimension mismatch")

    m = hidden.numel() // hidden_k
    n = weight.shape[0]
    k = hidden_k
    s = hidden_s

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
    linear_only_kernel[grid](
        hidden,
        weight,
        out,
        m,
        n,
        k,
        s,
        stride_a0,
        stride_a1,
        stride_a2,
        weight.stride(0),
        weight.stride(1),
        stride_o0,
        stride_o1,
        stride_o2,
    )
    return out


def _generic_mul(a, b):
    if a.numel() >= b.numel():
        x = a
        scale = b
    else:
        x = b
        scale = a
    out = torch.empty_like(x)
    n_elements = x.numel()
    channels = scale.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    broadcast_scale_mul_kernel[grid](x, scale, out, n_elements, channels)
    return out


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route == "fused_linear_mul":
        return _fused_linear_mul(args[0], args[1], args[2])
    if route == "broadcast_scale_mul":
        return _generic_mul(args[0], args[1])
    if route == "mmpose_outputs":
        return _generic_mul(args[1], args[2]), _linear_only(args[0], args[3], args[2])
    raise RuntimeError(f"Unknown optimization route: {route}")


def replacement_func():
    return shared_dispatch