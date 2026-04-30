import torch
import triton
import triton.language as tl


# ── Mean reduction kernel (2D coalesced, c as inner dim) ─────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64},  num_warps=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 128}, num_warps=16),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=16),
    ],
    key=['B', 'N', 'C'],
)
@triton.jit
def _mean_neg2_kernel(
    input_ptr, output_ptr, B, N, C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = tl.arange(0, BLOCK_N)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    n_mask = n_offs < N
    c_mask = c_offs < C

    # Load [BLOCK_N, BLOCK_C]: inner dim = C (stride 1) → coalesced
    ptrs = (input_ptr
            + pid_b * N * C
            + n_offs[:, None] * C
            + c_offs[None, :])
    x = tl.load(ptrs,
                mask=n_mask[:, None] & c_mask[None, :],
                other=0.0).to(tl.float32)

    sums = tl.sum(x, axis=0)
    means = (sums / N.to(tl.float32)).to(x.dtype)

    out_ptrs = output_ptr + pid_b * C + c_offs
    tl.store(out_ptrs, means, mask=c_mask)


# ── Linear + bias kernel ───────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def _linear_bias_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N_out, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        a = tl.load(input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(weight_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))

    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(output_ptr + m_offs[:, None] * stride_om + n_offs[None, :],
             acc.to(output_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


# ── Wrapper: computes BOTH outputs in one Python call ────────────────────────

@torch.fx.wrap
def compute_linear_and_mean(bias, weight, x_in3):
    # Operation 1: linear  [B, K] @ [K, N_out] + [N_out]
    B, K = x_in3.shape
    N_out = weight.shape[0]
    linear_out = torch.empty((B, N_out), dtype=x_in3.dtype, device=x_in3.device)
    grid_l = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),
                           triton.cdiv(N_out, meta['BLOCK_N']))
    _linear_bias_kernel[grid_l](
        x_in3, weight, bias, linear_out,
        B, N_out, K,
        x_in3.stride(0), x_in3.stride(1),
        weight.stride(0), weight.stride(1),
        linear_out.stride(0),
    )

    # Operation 2: mean  [B, N, C] → [B, C] over last-but-one dim
    B2, N2, C = bias.shape  # shape = [B2, N2, C]
    mean_out = torch.empty((B2, C), dtype=bias.dtype, device=bias.device)
    grid_m = lambda meta: (B2, triton.cdiv(C, meta['BLOCK_C']))
    _mean_neg2_kernel[grid_m](bias, mean_out, B2, N2, C)

    return linear_out, mean_out


# ── Pattern: matches the ENTIRE computation graph ────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = in_3.mean(-2)
    return (tmp_2, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    # in_0=bias, in_1=weight, in_2=mean_input, in_3=seq_output
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return compute_linear_and_mean