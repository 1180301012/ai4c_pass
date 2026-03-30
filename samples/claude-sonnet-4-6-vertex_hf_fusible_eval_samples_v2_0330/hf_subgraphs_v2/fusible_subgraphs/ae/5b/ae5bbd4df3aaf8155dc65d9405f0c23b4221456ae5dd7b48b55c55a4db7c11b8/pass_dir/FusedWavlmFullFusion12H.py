"""
Full fusion pass for WavLM GRU relative position computation - 12 heads.

Fuses ENTIRE computation: linear → view(1,12,199,2,4) → sum(-1) → sigmoid
  → chunk → elementwise ops → view(1,12,-1,1)

Key optimization: wsum trick reduces 8 dot products to 2.
  wsum0[k] = w[0,k]+w[1,k]+w[2,k]+w[3,k]  (pre-summed weight rows 0-3)
  wsum1[k] = w[4,k]+w[5,k]+w[6,k]+w[7,k]  (pre-summed weight rows 4-7)
  group_sum_0(t) = dot(in3[h,t,:], wsum0) + bsum0
  group_sum_1(t) = dot(in3[h,t,:], wsum1) + bsum1

Memory access: in3 [H,T,K] loaded as [BLOCK_T, K] per CTA → perfectly coalesced
(consecutive T positions × consecutive K values per thread).
"""
import torch
import triton
import triton.language as tl

_BLOCK_T   = 32
_NUM_WARPS = 1
_K         = 64   # input feature dim (always 64 for WavLM)


@triton.jit
def _wavlm_full_fusion_kernel(
    in3_ptr,    # [H, T, K] contiguous
    ws0_ptr,    # [K]  pre-summed weight rows 0-3
    ws1_ptr,    # [K]  pre-summed weight rows 4-7
    in2_ptr,    # [H]  per-head constant
    out_ptr,    # [H, T] output
    T,
    bsum0,      # python scalar: b[0]+b[1]+b[2]+b[3]
    bsum1,      # python scalar: b[4]+b[5]+b[6]+b[7]
    K: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h     = tl.program_id(0)
    t_blk = tl.program_id(1)
    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask   = t_offs < T

    in2_val = tl.load(in2_ptr + h).to(tl.float32)

    # Load wsum vectors [K] – small, cached in L2 after first CTA
    k_offs = tl.arange(0, K)
    ws0 = tl.load(ws0_ptr + k_offs).to(tl.float32)   # [K]
    ws1 = tl.load(ws1_ptr + k_offs).to(tl.float32)   # [K]

    # Coalesced 2D load: [BLOCK_T, K] – thread i owns row i (K consecutive fp16)
    in3_offs = t_offs[:, None] * K + k_offs[None, :]   # [BLOCK_T, K]
    in3_blk  = tl.load(
        in3_ptr + h * T * K + in3_offs,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)   # [BLOCK_T, K]

    # 2 dot products (in-register, no cross-thread comms)
    acc0 = tl.sum(in3_blk * ws0[None, :], axis=1) + bsum0   # [BLOCK_T]
    acc1 = tl.sum(in3_blk * ws1[None, :], axis=1) + bsum1   # [BLOCK_T]

    s0     = tl.sigmoid(acc0)
    s1     = tl.sigmoid(acc1)
    result = s0 * (s1 * in2_val - 1.0) + 2.0

    if IS_BF16:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.float16),  mask=mask)


@torch.fx.wrap
def wavlm_full_fusion_impl_12h(in_0, in_1, in_2, in_3):
    """
    in_0: bias   [8]
    in_1: weight [8, 64]
    in_2: const  [1, 12, 1, 1]
    in_3: input  [1, 12, 199, 64]
    returns: [1, 12, 199, 1]
    """
    H, T, K = 12, 199, _K
    dtype   = in_3.dtype
    is_bf16 = (dtype == torch.bfloat16)

    # Pre-compute summed weight rows (wsum trick: 8 dot products → 2)
    w_fp32 = in_1.to(torch.float32)
    ws0 = w_fp32[:4, :].sum(dim=0).to(dtype).contiguous()
    ws1 = w_fp32[4:, :].sum(dim=0).to(dtype).contiguous()
    b_fp32 = in_0.to(torch.float32)
    bsum0  = float(b_fp32[:4].sum().item())
    bsum1  = float(b_fp32[4:].sum().item())

    out      = torch.empty(H, T, dtype=dtype, device=in_3.device)
    in2_flat = in_2.reshape(H)
    in3_cont = in_3.reshape(H, T, K).contiguous()

    grid = (H, (T + _BLOCK_T - 1) // _BLOCK_T)
    _wavlm_full_fusion_kernel[grid](
        in3_cont, ws0, ws1, in2_flat, out,
        T, bsum0, bsum1,
        K=K,
        IS_BF16=is_bf16,
        BLOCK_T=_BLOCK_T,
        num_warps=_NUM_WARPS,
    )
    return out.reshape(1, H, T, 1)


# ─── Pattern / replacement API ────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4  = linear.view(1, 12, 199, 2, 4)
    tmp_5  = tmp_4.sum(-1, keepdim=False)
    tmp_6  = torch.sigmoid(tmp_5)
    chunk  = tmp_6.chunk(2, dim=-1)
    tmp_8  = chunk[0]
    tmp_9  = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return wavlm_full_fusion_impl_12h