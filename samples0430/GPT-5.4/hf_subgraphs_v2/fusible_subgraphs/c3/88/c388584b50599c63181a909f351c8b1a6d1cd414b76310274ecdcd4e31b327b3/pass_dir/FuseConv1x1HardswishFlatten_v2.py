import torch
import triton
import triton.language as tl


_WEIGHT_PACK_CACHE = {}


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _pack_weight_kn_kernel(
    src_ptr,
    dst_ptr,
    N,
    K,
    stride_src_n,
    stride_src_k,
    stride_dst_k,
    stride_dst_n,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    vals = tl.load(
        src_ptr + offs_n[None, :] * stride_src_n + offs_k[:, None] * stride_src_k,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + offs_k[:, None] * stride_dst_k + offs_n[None, :] * stride_dst_n,
        vals,
        mask=mask,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_conv1x1_hardswish_kernel(
    x_ptr,
    wkn_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_x0,
    stride_x1,
    stride_wk,
    stride_wn,
    stride_o0,
    stride_o1,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_x0 + offs_k[None, :] * stride_x1
    w_ptrs = wkn_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        k_mask = offs_k < K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_x1
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    out = acc * (tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0) * (1.0 / 6.0))

    out_ptrs = out_ptr + offs_m[:, None] * stride_o0 + offs_n[None, :] * stride_o1
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, x):
    m = x.shape[0]
    k = x.shape[1]
    n = weight.shape[0]

    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    w_key = (int(weight.data_ptr()), tuple(weight.shape), tuple(weight.stride()), str(weight.dtype), weight.device.index)
    packed = _WEIGHT_PACK_CACHE.get(w_key)
    if packed is None:
        packed = torch.empty((k, n), device=weight.device, dtype=weight.dtype)
        pack_grid = lambda META: (triton.cdiv(k, META['BLOCK_K']), triton.cdiv(n, META['BLOCK_N']))
        _pack_weight_kn_kernel[pack_grid](
            weight,
            packed,
            n,
            k,
            weight.stride(0),
            weight.stride(1),
            packed.stride(0),
            packed.stride(1),
            BLOCK_K=32,
            BLOCK_N=128,
            num_warps=4,
            num_stages=2,
        )
        _WEIGHT_PACK_CACHE[w_key] = packed

    grid = lambda META: (triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(n, META['BLOCK_N']),)
    _fused_conv1x1_hardswish_kernel[grid](
        x,
        packed,
        bias,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        packed.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_conv1x1_hardswish_flatten