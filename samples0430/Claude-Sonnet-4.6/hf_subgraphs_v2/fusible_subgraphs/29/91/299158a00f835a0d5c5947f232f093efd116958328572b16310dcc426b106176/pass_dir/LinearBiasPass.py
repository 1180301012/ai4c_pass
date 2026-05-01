import torch
import triton
import triton.language as tl


# K=448 always; pad up to the next power-of-two so we can cover K in one tile.
_BLOCK_K = 512   # 512 > 448; elements 448..511 are masked out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 16},  num_warps=4),
        triton.Config({'BLOCK_B': 32},  num_warps=4),
        triton.Config({'BLOCK_B': 64},  num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 16},  num_warps=8),
        triton.Config({'BLOCK_B': 32},  num_warps=8),
        triton.Config({'BLOCK_B': 64},  num_warps=8),
        triton.Config({'BLOCK_B': 128}, num_warps=8),
    ],
    key=['B', 'N', 'K'],
)
@triton.jit
def _linear_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, N, K,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute out[b, n] = dot(x[b, :], w[n, :]) + bias[n]
    Grid: (ceil(B / BLOCK_B), N)
    """
    b_block = tl.program_id(0)
    n       = tl.program_id(1)

    b_start   = b_block * BLOCK_B
    b_offsets = b_start + tl.arange(0, BLOCK_B)
    b_mask    = b_offsets < B

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask    = k_offsets < K

    # Load weight row for this output neuron (1 × K)
    w = tl.load(w_ptr + n * K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

    # Load input tile (BLOCK_B × K)
    x_ptrs = x_ptr + b_offsets[:, None] * K + k_offsets[None, :]
    x      = tl.load(x_ptrs,
                     mask=b_mask[:, None] & k_mask[None, :],
                     other=0.0).to(tl.float32)

    # Dot product: (BLOCK_B,)
    acc = tl.sum(x * w[None, :], axis=1)

    # Bias
    bias_val = tl.load(b_ptr + n).to(tl.float32)
    acc += bias_val

    # Store – cast back to the input dtype
    out_ptrs = out_ptr + b_offsets * N + n
    tl.store(out_ptrs, acc.to(x_ptr.dtype.element_ty), mask=b_mask)


@torch.fx.wrap
def triton_linear_bias(x, weight, bias):
    # x: (B, K),  weight: (N, K),  bias: (N,)  →  out: (B, N)
    B = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]

    out  = torch.empty((B, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']), N)

    _linear_bias_kernel[grid](
        x, weight, bias, out,
        B, N, K,
        BLOCK_K=_BLOCK_K,
    )

    return out


# ── Pass interface ────────────────────────────────────────────────────────────

def pattern(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def replacement_args(x, w, b):
    return (x, w, b)


def replacement_func():
    return triton_linear_bias