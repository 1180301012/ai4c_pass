"""
Pass: Fuse linear + dropout(any p, training=False) + transpose(1,2)
Returns: (transposed_out, linear_out)   ← swapped order

Covers all models with return order (transposed, linear):
  - hf-tiny UniSpeechSat float16   1248x32->16  p=0.0
  - hf-tiny UniSpeechSat bfloat16  1248x32->16  p=0.0

Strategy:
  - Triton kernel: GEMM + bias only (no transposed write)
  - Return out.transpose(1,2) as FIRST output (free view)
  - Wildcard p matches any dropout probability
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32,  'BLOCK_H': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32,  'BLOCK_H': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_S': 32,  'BLOCK_H': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16,  'BLOCK_H': 16,  'BLOCK_K': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_S': 64,  'BLOCK_H': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['B', 'S', 'H', 'K'],
)
@triton.jit
def _linear_bias_kernel_t_out(
    x_ptr, w_ptr, bias_ptr,
    out_ptr,
    B, S, H, K,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute out[B,S,H] = x[B,S,K] @ w[H,K].T + bias[H]."""
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

        x_mask = (s_range[:, None] < S) & (k_range[None, :] < K)
        x = tl.load(
            x_ptr + pid_b * S * K + s_range[:, None] * K + k_range[None, :],
            mask=x_mask, other=0.0
        )

        w_mask = (h_range[:, None] < H) & (k_range[None, :] < K)
        w = tl.load(
            w_ptr + h_range[:, None] * K + k_range[None, :],
            mask=w_mask, other=0.0
        )

        acc = tl.dot(x, tl.trans(w), acc)

    # Add bias and store out [B, S, H]
    bias = tl.load(bias_ptr + h_range, mask=h_range < H, other=0.0)
    acc += bias[None, :].to(tl.float32)

    out_mask = (s_range[:, None] < S) & (h_range[None, :] < H)
    tl.store(
        out_ptr + pid_b * S * H + s_range[:, None] * H + h_range[None, :],
        acc.to(OUTPUT_DTYPE),
        mask=out_mask
    )


@torch.fx.wrap
def _triton_linear_t_out(bias, weight, x):
    """Opaque FX node: Triton linear+bias, returns out [B,S,H]."""
    B, S, K = x.shape
    H = weight.shape[0]
    out = torch.empty(B, S, H, dtype=x.dtype, device=x.device)

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

    _linear_bias_kernel_t_out[grid](
        x, weight, bias,
        out,
        B, S, H, K,
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )
    return out


# NOT wrapped — FX traces into this so the TWO return values become two separate
# FX nodes, satisfying: len(copied_returning_nodes) == len(match.returning_nodes) == 2
def triton_linear_transpose_t_out(bias, weight, x):
    """Returns (transposed_out [B,H,S], linear_out [B,S,H])."""
    out = _triton_linear_t_out(bias, weight, x)
    return out.transpose(1, 2), out   # transpose first (free view)


# Wildcard p: matches any dropout probability
def pattern(in_0, in_1, in_2, p):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, p, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2, p):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_linear_transpose_t_out