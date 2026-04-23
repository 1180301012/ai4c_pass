import torch
import triton
import triton.language as tl


_WEIGHT_T_CACHE = {}


def _get_weight_t(weight: torch.Tensor) -> torch.Tensor:
    key = (
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        str(weight.device),
    )
    cached = _WEIGHT_T_CACHE.get(key)
    if cached is None:
        cached = weight.transpose(0, 1).contiguous()
        _WEIGHT_T_CACHE[key] = cached
    return cached


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def linear_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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
    k_iter = 0
    while k_iter < K:
        k_offsets = k_iter + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = acc + tl.dot(a, b)
        k_iter += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@torch.fx.wrap
def triton_linear_448_1536_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    # Shape-specialized for this EfficientFormer qkv projection.
    K = 448
    N = 1536
    if x.shape[-1] != K or weight.shape != (N, K) or bias.shape != (N,):
        return torch.nn.functional.linear(x, weight, bias)

    m = x.numel() // K
    if x.dtype == torch.float32 or m < 1024:
        return torch.nn.functional.linear(x, weight, bias)

    x_2d = x.reshape(m, K)
    wt = _get_weight_t(weight)
    out = torch.empty((m, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    linear_bias_kernel[grid](
        x_2d,
        wt,
        bias,
        out,
        m,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        wt.stride(0),
        wt.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out.reshape(*x.shape[:-1], N)