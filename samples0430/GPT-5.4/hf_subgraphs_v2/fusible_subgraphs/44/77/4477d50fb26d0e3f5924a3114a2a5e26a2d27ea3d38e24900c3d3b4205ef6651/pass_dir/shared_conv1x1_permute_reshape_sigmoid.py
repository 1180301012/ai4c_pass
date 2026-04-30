import torch
import triton
import triton.language as tl


C_IN = 512


@triton.jit
def _fused_conv1x1_sigmoid_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    m_size,
    hw_size,
    out_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    mask_m = offs_m < m_size
    mask_n = offs_n < out_c

    hw = offs_m % hw_size
    n = offs_m // hw_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_IN, BLOCK_K):
        offs_k = k_start + offs_k_base
        x_ptrs = x_ptr + (n[:, None] * C_IN + offs_k[None, :]) * hw_size + hw[:, None]
        w_ptrs = w_ptr + offs_n[None, :] * C_IN + offs_k[:, None]

        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = 1.0 / (1.0 + tl.exp(-acc))

    out_ptrs = out_ptr + offs_m[:, None] * out_c + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_permute_reshape_sigmoid(bias, weight, x):
    batch = x.shape[0]
    hw = x.shape[2] * x.shape[3]
    out_c = weight.shape[0]
    m_size = batch * hw

    out = torch.empty((m_size, out_c), device=x.device, dtype=x.dtype)

    is_fp32 = x.dtype == torch.float32
    if out_c == 1:
        block_m = 128 if is_fp32 else 256
        block_n = 1
        block_k = 32 if is_fp32 else 64
        num_warps = 4
    elif out_c <= 4:
        block_m = 128 if is_fp32 else 256
        block_n = 4
        block_k = 32 if is_fp32 else 64
        num_warps = 4 if is_fp32 else 8
    elif out_c <= 9:
        block_m = 128
        block_n = 16
        block_k = 32 if is_fp32 else 64
        num_warps = 8
    else:
        block_m = 64
        block_n = 32
        block_k = 32 if is_fp32 else 64
        num_warps = 8

    grid = (
        triton.cdiv(m_size, block_m),
        triton.cdiv(out_c, block_n),
    )

    _fused_conv1x1_sigmoid_kernel[grid](
        x,
        weight,
        bias,
        out,
        m_size,
        hw,
        out_c,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return out.reshape(batch, hw, out_c)