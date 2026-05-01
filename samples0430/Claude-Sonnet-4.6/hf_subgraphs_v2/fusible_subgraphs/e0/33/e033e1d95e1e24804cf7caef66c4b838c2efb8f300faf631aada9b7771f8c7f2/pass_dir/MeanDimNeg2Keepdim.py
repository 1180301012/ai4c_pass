import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32,  'BLOCK_M': 256}, num_warps=4),
        triton.Config({'BLOCK_K': 32,  'BLOCK_M': 512}, num_warps=4),
        triton.Config({'BLOCK_K': 64,  'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 64,  'BLOCK_M': 256}, num_warps=4),
        triton.Config({'BLOCK_K': 64,  'BLOCK_M': 512}, num_warps=8),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 64},  num_warps=4),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 256}, num_warps=8),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 32},  num_warps=4),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 64},  num_warps=4),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 128}, num_warps=8),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 256}, num_warps=8),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 512}, num_warps=8),
        triton.Config({'BLOCK_K': 256, 'BLOCK_M': 1024}, num_warps=16),
    ],
    key=['N', 'M', 'K'],
)
@triton.jit
def _mean_dim_neg2_keepdim_kernel(
    input_ptr,
    output_ptr,
    N, M, K,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Reduce input[N, M, K] → output[N, 1, K] by computing mean over dim 1 (the M dim).
    Grid: (N, cdiv(K, BLOCK_K))
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # K-dimension offsets handled by this program
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Accumulate sum over M in float32 for numerical stability
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    base = input_ptr + pid_n * M * K

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M

        # Pointer block: shape [BLOCK_M, BLOCK_K], contiguous in the K dimension
        ptrs = base + m_offs[:, None] * K + k_offs[None, :]
        mask_2d = m_mask[:, None] & k_mask[None, :]

        vals = tl.load(ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)  # reduce over BLOCK_M rows → [BLOCK_K]

    mean_vals = acc / M  # still float32

    # Store – Triton auto-converts float32 → output dtype (fp16/bf16/fp32)
    out_ptrs = output_ptr + pid_n * K + k_offs
    tl.store(out_ptrs, mean_vals, mask=k_mask)


@torch.fx.wrap
def triton_mean_dim_neg2_keepdim(x):
    """
    Drop-in replacement for x.mean(dim=-2, keepdim=True) on a 3-D tensor.
    Input:  x  – [N, M, K]
    Output: out – [N, 1, K]  (same dtype as x)
    """
    N = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]

    output = torch.empty((N, 1, K), dtype=x.dtype, device=x.device)

    grid = lambda meta: (N, triton.cdiv(K, meta['BLOCK_K']))
    _mean_dim_neg2_keepdim_kernel[grid](x, output, N, M, K)
    return output


# ---------------------------------------------------------------------------
# Pattern / replacement API expected by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(x):
    return x.mean(dim=-2, keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_dim_neg2_keepdim