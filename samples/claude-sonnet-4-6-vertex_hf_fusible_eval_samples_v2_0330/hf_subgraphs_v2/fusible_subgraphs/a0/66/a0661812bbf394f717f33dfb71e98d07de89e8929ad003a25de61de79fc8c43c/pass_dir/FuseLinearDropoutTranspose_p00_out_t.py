"""
Pass: Fuse linear + dropout(p=0.0, training=False) + transpose(1,2)
Returns: (linear_out, transposed_out)

Pattern matched: anton-l_distilhubert-ft-common-language (float32, 249x512->768)
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16, 'BLOCK_H': 16, 'BLOCK_K': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['B', 'S', 'H', 'K'],
)
@triton.jit
def _linear_bias_transpose_kernel_p00_out_t(
    x_ptr, w_ptr, bias_ptr,
    out_ptr, out_t_ptr,
    B, S, H, K,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    h_start = pid_h * BLOCK_H

    s_range = s_start + tl.arange(0, BLOCK_S)
    h_range = h_start + tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)

        # Load X tile [BLOCK_S, BLOCK_K] from [B, S, K]
        x_mask = (s_range[:, None] < S) & (k_range[None, :] < K)
        x = tl.load(
            x_ptr + pid_b * S * K + s_range[:, None] * K + k_range[None, :],
            mask=x_mask, other=0.0
        )

        # Load W tile [BLOCK_H, BLOCK_K] from [H, K]
        w_mask = (h_range[:, None] < H) & (k_range[None, :] < K)
        w = tl.load(
            w_ptr + h_range[:, None] * K + k_range[None, :],
            mask=w_mask, other=0.0
        )

        # acc [BLOCK_S, BLOCK_H] += x @ w.T
        acc = tl.dot(x, tl.trans(w), acc)

    # Add bias
    bias = tl.load(bias_ptr + h_range, mask=h_range < H, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Store out [B, S, H]
    out_mask = (s_range[:, None] < S) & (h_range[None, :] < H)
    tl.store(
        out_ptr + pid_b * S * H + s_range[:, None] * H + h_range[None, :],
        acc.to(OUTPUT_DTYPE),
        mask=out_mask
    )

    # Store out_t [B, H, S] — transposed layout
    out_t_mask = (h_range[:, None] < H) & (s_range[None, :] < S)
    tl.store(
        out_t_ptr + pid_b * H * S + h_range[:, None] * S + s_range[None, :],
        tl.trans(acc).to(OUTPUT_DTYPE),
        mask=out_t_mask
    )


@torch.fx.wrap
def _triton_linear_both_p00_out_t(bias, weight, x):
    """Opaque FX node: launches kernel, returns (linear_out, transposed_out)."""
    B, S, K = x.shape
    H = weight.shape[0]
    out = torch.empty(B, S, H, dtype=x.dtype, device=x.device)
    out_t = torch.empty(B, H, S, dtype=x.dtype, device=x.device)

    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    OUTPUT_DTYPE = dtype_map[x.dtype]

    grid = lambda meta: (
        triton.cdiv(S, meta['BLOCK_S']),
        triton.cdiv(H, meta['BLOCK_H']),
        B,
    )

    _linear_bias_transpose_kernel_p00_out_t[grid](
        x, weight, bias,
        out, out_t,
        B, S, H, K,
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )
    return out, out_t


# NOT wrapped — FX traces into this so getitem nodes appear as separate outputs.
def triton_linear_transpose_out_t_p00(bias, weight, x):
    result = _triton_linear_both_p00_out_t(bias, weight, x)
    return result[0], result[1]


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_linear_transpose_out_t_p00