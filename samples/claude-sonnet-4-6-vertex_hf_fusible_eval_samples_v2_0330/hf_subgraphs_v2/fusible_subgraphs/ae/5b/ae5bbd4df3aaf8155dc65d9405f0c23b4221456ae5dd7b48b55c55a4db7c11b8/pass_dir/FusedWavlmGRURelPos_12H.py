"""
H-independent fused pass for WavLM GRU relative position computation.

Works for BOTH wavlm_base (12 heads) and wavlm_large (16 heads).

Pattern matched: tmp_4.sum(-1) → sigmoid → chunk(2,dim=-1) → elementwise ops
  (tmp_4 is the [1,H,T,2,4] output from the H-specific linear.view)

Key optimizations:
- No @triton.autotune: avoids autotuning overhead during warmup/measurement
- 2D grid (H, ceil(T/BLOCK_T)): head from program_id(0), no integer division
- BLOCK_T=32, num_warps=1: 32 threads for 32 T positions → 100% utilisation,
  maximises block count (84 or 112 blocks) for better SM occupancy
- Fixed grid as tuple: no lambda overhead on hot path
- 8 loads + in-register sum: simple, register-efficient
"""
import torch
import triton
import triton.language as tl

_BLOCK_T  = 32
_NUM_WARPS = 1    # 1 warp = BLOCK_T threads → maximises block count for SM occupancy


@triton.jit
def _wavlm_sum_sigmoid_elem_kernel(
    tmp4_ptr,   # [1, H, T, 2, 4] contiguous → element [0,h,t,g,k] at h*T*8+t*8+g*4+k
    in2_ptr,    # [1, H, 1, 1]  contiguous   → element [0,h,0,0] at h
    out_ptr,    # [H, T]        contiguous   → element [h, t]    at h*T + t
    T,          # time steps (runtime int)
    IS_BF16: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h     = tl.program_id(0)
    t_blk = tl.program_id(1)
    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    mask   = t_offs < T

    # Per-head scalar constant
    in2_val = tl.load(in2_ptr + h).to(tl.float32)

    # 8 consecutive loads per thread; with BLOCK_T=32 (default 4 warps = 128 threads):
    # Triton distributes 32 logical elements across warps efficiently.
    base = h * T * 8 + t_offs * 8
    v0 = tl.load(tmp4_ptr + base + 0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(tmp4_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(tmp4_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(tmp4_ptr + base + 3, mask=mask, other=0.0).to(tl.float32)
    v4 = tl.load(tmp4_ptr + base + 4, mask=mask, other=0.0).to(tl.float32)
    v5 = tl.load(tmp4_ptr + base + 5, mask=mask, other=0.0).to(tl.float32)
    v6 = tl.load(tmp4_ptr + base + 6, mask=mask, other=0.0).to(tl.float32)
    v7 = tl.load(tmp4_ptr + base + 7, mask=mask, other=0.0).to(tl.float32)

    sum0 = v0 + v1 + v2 + v3
    sum1 = v4 + v5 + v6 + v7

    s0     = tl.sigmoid(sum0)
    s1     = tl.sigmoid(sum1)
    result = s0 * (s1 * in2_val - 1.0) + 2.0

    if IS_BF16:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + h * T + t_offs, result.to(tl.float16),  mask=mask)


@torch.fx.wrap
def wavlm_sum_sigmoid_elem_impl(tmp_4, in_2):
    """
    tmp_4: [1, H, T, 2, 4]
    in_2:  [1, H, 1, 1]
    returns: [1, H, T, 1]
    """
    H = tmp_4.shape[1]
    T = tmp_4.shape[2]
    out  = torch.empty(H, T, dtype=tmp_4.dtype, device=tmp_4.device)
    grid = (H, (T + _BLOCK_T - 1) // _BLOCK_T)
    _wavlm_sum_sigmoid_elem_kernel[grid](
        tmp_4, in_2, out, T,
        IS_BF16=(tmp_4.dtype == torch.bfloat16),
        BLOCK_T=_BLOCK_T,
        num_warps=_NUM_WARPS,
    )
    return out.reshape(1, H, T, 1)

# ─── Pattern / replacement API ────────────────────────────────────────────────

def pattern(tmp_4, in_2):
    """
    H-independent: tmp_4 is [1,H,T,2,4] regardless of H (12 or 16).
    The final view(1,H,-1,1) remains in the graph (no-op on [1,H,T,1]).
    """
    tmp_5  = tmp_4.sum(-1, keepdim=False)
    tmp_6  = torch.sigmoid(tmp_5)
    chunk  = tmp_6.chunk(2, dim=-1)
    tmp_8  = chunk[0]
    tmp_9  = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    return tmp_13


def replacement_args(tmp_4, in_2):
    return (tmp_4, in_2)


def replacement_func():
    return wavlm_sum_sigmoid_elem_impl