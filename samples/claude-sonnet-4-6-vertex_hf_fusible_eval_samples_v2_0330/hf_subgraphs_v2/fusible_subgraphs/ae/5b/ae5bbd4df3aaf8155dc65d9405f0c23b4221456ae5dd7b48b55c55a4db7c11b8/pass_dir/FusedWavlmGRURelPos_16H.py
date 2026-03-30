"""
Fused pass for WavLM GRU relative position computation with 16 attention heads.

Fuses: linear → view(1,16,199,2,4) → sum(-1) → sigmoid → chunk → elementwise ops → view

Key optimization: since the last two operations before sigmoid are:
  view [H, T, 8] → [H, T, 2, 4], then sum over last dim → [H, T, 2]
  sum_group_0 = out[0]+out[1]+out[2]+out[3]
  sum_group_1 = out[4]+out[5]+out[6]+out[7]
  where out[j] = dot(in3, w[j]) + b[j]
  We can precompute wsum0 = w[0]+w[1]+w[2]+w[3], wsum1 = w[4]+w[5]+w[6]+w[7]
  and compute only 2 dot products instead of 8.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 16}, num_warps=2),
        triton.Config({'BLOCK_T': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 128}, num_warps=8),
    ],
    key=['T', 'K'],
)
@triton.jit
def _fused_wavlm_gru_kernel_16h(
    in3_ptr,    # [H, T, K]  contiguous
    ws0_ptr,    # [K]  wsum0 = w[0]+w[1]+w[2]+w[3]
    ws1_ptr,    # [K]  wsum1 = w[4]+w[5]+w[6]+w[7]
    in2_ptr,    # [H]  per-head scalar (from [1,H,1,1])
    out_ptr,    # [H, T]  output
    H, T,
    bsum0,      # scalar: b[0]+b[1]+b[2]+b[3]
    bsum1,      # scalar: b[4]+b[5]+b[6]+b[7]
    K: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)
    t_blk = tl.program_id(1)
    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offs < T
    k_offs = tl.arange(0, K)

    # Load pre-summed weights [K]
    ws0 = tl.load(ws0_ptr + k_offs).to(tl.float32)
    ws1 = tl.load(ws1_ptr + k_offs).to(tl.float32)

    # Load in3 block [BLOCK_T, K]
    in3_blk = tl.load(
        in3_ptr + h * T * K + t_offs[:, None] * K + k_offs[None, :],
        mask=mask_t[:, None],
        other=0.0,
    ).to(tl.float32)

    # Two dot products: sum(in3 * ws0) + bsum0, sum(in3 * ws1) + bsum1
    dot0 = tl.sum(in3_blk * ws0[None, :], axis=1) + bsum0   # [BLOCK_T]
    dot1 = tl.sum(in3_blk * ws1[None, :], axis=1) + bsum1   # [BLOCK_T]

    # Sigmoid of the two sums
    s0 = tl.sigmoid(dot0)
    s1 = tl.sigmoid(dot1)

    # Load per-head constant from in_2 [1, H, 1, 1] flattened to [H]
    in2_val = tl.load(in2_ptr + h).to(tl.float32)

    # Element-wise: s0 * (s1 * in2 - 1.0) + 2.0
    result = s0 * (s1 * in2_val - 1.0) + 2.0   # [BLOCK_T]

    # Store with correct dtype
    if IS_BF16:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.bfloat16), mask=mask_t)
    else:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.float16), mask=mask_t)


@torch.fx.wrap
def fused_wavlm_gru_impl_16h(in_0, in_1, in_2, in_3):
    """
    in_0: bias   [8]
    in_1: weight [8, 64]
    in_2: const  [1, 16, 1, 1]
    in_3: input  [1, 16, 199, 64]
    returns: (output,) with output shape [1, 16, 199, 1]
    """
    H, T, K = 16, 199, 64
    dtype = in_3.dtype
    device = in_3.device
    is_bf16 = (dtype == torch.bfloat16)

    # Pre-compute summed weight rows to reduce dot products from 8 to 2
    w_fp32 = in_1.to(torch.float32).contiguous()   # [8, 64]
    ws0 = w_fp32[:4, :].sum(dim=0).to(dtype).contiguous()  # [64]
    ws1 = w_fp32[4:, :].sum(dim=0).to(dtype).contiguous()  # [64]

    b_fp32 = in_0.to(torch.float32)
    bsum0 = float(b_fp32[:4].sum().item())
    bsum1 = float(b_fp32[4:].sum().item())

    out = torch.empty(1, H, T, 1, dtype=dtype, device=device)
    in2_flat = in_2.reshape(H).contiguous()
    in3_cont = in_3.reshape(H, T, K).contiguous()
    out_flat = out.reshape(H, T)

    grid = lambda meta: (H, triton.cdiv(T, meta['BLOCK_T']))
    _fused_wavlm_gru_kernel_16h[grid](
        in3_cont, ws0, ws1, in2_flat, out_flat,
        H, T, bsum0, bsum1,
        K=K,
        IS_BF16=is_bf16,
    )

    return (out,)


# ─── Pattern / replacement API ───────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 16, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 16, -1, 1)
    return (tmp_14,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_wavlm_gru_impl_16h