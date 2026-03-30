import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small-M configs (BigBird: M=17, N=3072, K=768)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        # Square-M configs (RECT_L: M=128, N=128, K=128)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_kernel_bigbird(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM + bias kernel.
    Computes: out[m, n] = sum_k x[m, k] * W[n, k] + bias[n]
    W is stored as [N, K]; we access the transpose by swapping index strides.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)

        # Load x block: [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_off[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        )

        # Load W as [BLOCK_K, BLOCK_N] (transposed access: w[k, n] = W[n, k])
        w_mask = (k_off[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(
            w_ptr + k_off[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=w_mask, other=0.0,
        )

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    # Store with appropriate dtype
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if IS_BF16:
        tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptr + out_offs, acc.to(tl.float16), mask=out_mask)


@torch.fx.wrap
def triton_dropout_linear_bigbird(bias, weight, x):
    """
    Replacement for: dropout(x, 0.1, training=False) -> linear(x, weight, bias)
    Since dropout has training=False, it is identity. We fuse to a direct GEMM+bias.
    """
    if not x.is_cuda:
        x = x.cuda()
    weight = weight.to(x.device).contiguous()
    bias = bias.to(x.device).contiguous()

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    is_bf16 = 1 if x.dtype == torch.bfloat16 else 0

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _fused_linear_kernel_bigbird[grid](
        x_2d, weight, bias, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        IS_BF16=is_bf16,
    )

    return out.reshape(*orig_shape[:-1], N)


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_dropout_linear_bigbird