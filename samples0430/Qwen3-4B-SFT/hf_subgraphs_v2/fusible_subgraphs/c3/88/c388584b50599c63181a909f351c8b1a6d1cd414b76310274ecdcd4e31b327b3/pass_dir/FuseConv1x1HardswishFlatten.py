import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_M=32: covers M=32 in 1 block, all proven best performers
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4, 'num_stages': 6, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4, 'num_stages': 6, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 6, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 6, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 5, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_conv1x1_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # L2-swizzled block-ID mapping (Triton matmul tutorial pattern)
    # This groups consecutive pids that share the same X (M-tile) for better L2 broadcast.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        # x  [BLOCK_M, BLOCK_K]: coalesced inner dim K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # w  [BLOCK_N, BLOCK_K]: coalesced inner dim K, then transpose for tl.dot
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    # Bias + HardSwish
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc  = acc + bias[None, :]
    temp = acc + 3.0
    temp = tl.minimum(tl.maximum(temp, 0.0), 6.0)
    result = acc * temp * (1.0 / 6.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(in_0, in_1, in_2):
    _batch  = in_2.shape[0]
    K       = in_2.shape[1]
    _h      = in_2.shape[2]
    _w      = in_2.shape[3]
    stride_xm = _h * _w
    stride_xk = 1
    out_channels = in_1.shape[0]

    out = torch.empty((_batch, out_channels), dtype=in_2.dtype, device=in_2.device)

    # 1-D grid: allows GROUP_SIZE_M-based swizzle
    grid = lambda meta: (
        triton.cdiv(_batch, meta['BLOCK_M']) * triton.cdiv(out_channels, meta['BLOCK_N']),
    )

    _fused_conv1x1_hardswish_kernel[grid](
        in_2, in_1, in_0, out,
        _batch, out_channels, K,
        stride_xm, stride_xk,
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
    )

    return out


# ---------- Pattern / replacement hooks ----------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_hardswish_flatten