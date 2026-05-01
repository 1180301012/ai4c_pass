import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.ops.aten.convolution.default(in_2, in_1, in_0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    sig = torch.ops.aten.sigmoid.default(conv2d)
    tmp_3 = torch.ops.aten.mul.Tensor(conv2d, sig)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['GEMM_M', 'GEMM_N', 'GEMM_K'],
)
@triton.jit
def _conv1x1_bias_silu_kernel_v2(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    GEMM_M, GEMM_N, GEMM_K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(GEMM_K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < GEMM_M) & (offs_k[None, :] < GEMM_K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < GEMM_K) & (offs_n[None, :] < GEMM_N), other=0.0)
        acc += tl.dot(a, b)
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < GEMM_M).to(tl.float32)
    acc = acc + bias[:, None]
    acc = acc * tl.sigmoid(acc)
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c,
             mask=(offs_m[:, None] < GEMM_M) & (offs_n[None, :] < GEMM_N))


@torch.fx.wrap
def conv1x1_bias_silu_v2(bias, weight, x):
    N_batch, C_in, H, W = x.shape
    C_out = weight.shape[0]
    spatial = N_batch * H * W
    output = torch.empty((N_batch, C_out, H, W), dtype=x.dtype, device=x.device)
    stride_am = weight.stride(0)
    stride_ak = weight.stride(1)
    stride_bk = x.stride(1)
    stride_bn = x.stride(3)
    stride_cm = H * W
    stride_cn = 1
    grid = lambda meta: (triton.cdiv(C_out, meta['BLOCK_M']), triton.cdiv(spatial, meta['BLOCK_N']))
    _conv1x1_bias_silu_kernel_v2[grid](
        weight, x, bias, output,
        C_out, spatial, C_in,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return output


def replacement_func():
    return conv1x1_bias_silu_v2