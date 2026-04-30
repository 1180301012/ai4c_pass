import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def linear_n2_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Specialized kernel for small N (N=2)
    # Each program computes BLOCK_M rows of the output, all N columns
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < M

    # Accumulators for N=2 output values per row
    acc0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        off_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = off_k < K

        # Load x: [BLOCK_M, BLOCK_K]
        x = tl.load(x_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk,
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load weight row 0: W[0, k_block] = [BLOCK_K]
        w0 = tl.load(weight_ptr + 0 * stride_wn + off_k * stride_wk,
                     mask=mask_k, other=0.0)

        # Load weight row 1: W[1, k_block] = [BLOCK_K]
        w1 = tl.load(weight_ptr + 1 * stride_wn + off_k * stride_wk,
                     mask=mask_k, other=0.0)

        # Compute partial dot products using element-wise multiply + reduce
        # This avoids tl.dot overhead for small N
        acc0 += tl.sum(x.to(tl.float32) * w0.to(tl.float32)[None, :], axis=1)
        acc1 += tl.sum(x.to(tl.float32) * w1.to(tl.float32)[None, :], axis=1)

    # Add bias
    b0 = tl.load(bias_ptr + 0).to(tl.float32)
    b1 = tl.load(bias_ptr + 1 * stride_on).to(tl.float32)
    acc0 += b0
    acc1 += b1

    # Store output: y[m, 0] and y[m, 1]
    tl.store(out_ptr + off_m * stride_om + 0 * stride_on, acc0, mask=mask_m)
    tl.store(out_ptr + off_m * stride_om + 1 * stride_on, acc1, mask=mask_m)


@torch.fx.wrap
def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    linear_n2_kernel[grid](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
        M=M, N=N, K=K,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_wn=weight.stride(0), stride_wk=weight.stride(1),
        stride_om=out.stride(0), stride_on=out.stride(1),
    )

    return out


def replacement_func():
    return triton_linear