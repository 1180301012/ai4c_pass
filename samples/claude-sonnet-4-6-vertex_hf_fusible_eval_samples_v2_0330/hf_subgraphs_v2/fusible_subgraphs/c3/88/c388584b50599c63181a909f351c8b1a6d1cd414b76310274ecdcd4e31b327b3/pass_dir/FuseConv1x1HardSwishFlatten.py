import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d  +  hardswish (inplace)  +  flatten(1,-1)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    # in_0 = bias  [C_out]
    # in_1 = weight [C_out, C_in, 1, 1]
    # in_2 = input  [N, C_in, 1, 1]
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: GEMM + bias + HardSwish, output written in original dtype
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Lean set – powers of 2 only, covers both batch=1 and batch=32
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_hardswish_kernel(
    input_ptr,   # [M, K]  – in_2 reshaped
    weight_ptr,  # [N, K]  – in_1 reshaped
    bias_ptr,    # [N]     – in_0
    output_ptr,  # [M, N]  – result (native dtype)
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K

        # A tile: input[m, k]  -> shape [BLOCK_M, BLOCK_K]
        a = tl.load(
            input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # B tile: weight[n, k] transposed -> b[k, n]  shape [BLOCK_K, BLOCK_N]
        b = tl.load(
            weight_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )

        acc += tl.dot(a, b)

    # Bias add (broadcast over M)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    shifted  = acc + 3.0
    clamped  = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    out_f32  = acc * clamped * (1.0 / 6.0)

    # Cast back to the original dtype and store
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        out_f32.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# Python-side wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def conv1x1_hardswish_flatten(bias, weight, x):
    """
    Fused 1x1-conv + HardSwish + flatten.
    bias   : [C_out]
    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, 1, 1]
    returns: ([N, C_out],)
    """
    N    = x.shape[0]
    C_in = x.shape[1]
    C_out = weight.shape[0]

    # Treat 1x1 spatial as a flat GEMM
    x_2d = x.view(N, C_in)          # [N, C_in]
    w_2d = weight.view(C_out, C_in)  # [C_out, C_in]

    output = torch.empty((N, C_out), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_M']),
        triton.cdiv(C_out, META['BLOCK_N']),
    )

    _conv1x1_hardswish_kernel[grid](
        x_2d, w_2d, bias, output,
        N, C_out, C_in,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        output.stride(0), output.stride(1),
    )

    return (output,)


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------

def replacement_func():
    return conv1x1_hardswish_flatten