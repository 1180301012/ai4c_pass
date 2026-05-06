"""
Shared fused Triton kernel: softmax(dim=2) + weighted mean reduction + cat.

Fuses: softmax(x, dim=2)  ->  reshape(-1,17,64,64)  ->  mul(in_0)/mul(in_1)
       ->  reshape(B,17,-1) -> sum(dim=2,keepdim=True)
       ->  cat([sum_x, sum_y], dim=-1)

One Triton program handles one (batch, head) pair of 4096 spatial elements.
Softmax in fp32 for numerical stability; outputs written in original dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_weightedmean_kernel(
    logits_ptr,   # [N, 4096]  contiguous, fp16/bf16/fp32
    x_ptr,        # [1, 1, 1, 64]  contiguous, fp16/bf16/fp32
    y_ptr,        # [1, 1, 64, 1]  contiguous, fp16/bf16/fp32
    out_ptr,      # [N, 17, 64, 64]  contiguous, original dtype
    sum_out_ptr,  # [N, 17, 128]    contiguous, original dtype
    BATCH: tl.constexpr,
):
    """
    N = BATCH * H,  H = 17,  L = 4096
    Each program id represents one (b, h) pair.
    """
    pid  = tl.program_id(0)
    b    = pid // 17
    h    = pid  % 17
    slot = tl.arange(0, 64)
    j    = tl.arange(0, 4096)

    # ── Load logits and compute softmax (fp32) ──────────────────────────────
    base = b * 17 * 4096 + h * 4096
    x_vals = tl.load(logits_ptr + base + j).to(tl.float32)
    # numerically stable softmax
    x_max      = tl.max(x_vals, axis=0)
    x_exp      = tl.exp(x_vals - x_max)
    x_sum      = tl.sum(x_exp, axis=0)
    x_softmax  = x_exp / x_sum                   # [4096]
    x_invlogit = 1.0 / (1.0 + tl.exp(-x_vals))  # [4096]

    # ── Load x[0,0,0,:] and y[0,0,:,0] (both shape 64) ─────────────────────
    x_val  = tl.load(x_ptr).to(tl.float32)   # [64]
    y_val  = tl.load(y_ptr).to(tl.float32)   # [64]

    # ── Weighted mean over slot × spatial (one value per slot) ──────────────
    # slot s → which (i, s_rem) in spatial dim: j = i*64 + s_rem
    # x mean: sum_{i,s_rem} x_val[i]*softmax[j]  / 4096  for each slot s_rem
    x_mean = tl.zeros([64], dtype=tl.float32)
    y_mean = tl.zeros([64], dtype=tl.float32)

    for i in range(64):
        j_start   = i * 64
        x_local   = x_val[i]               # scalar
        y_local   = y_val[i]               # scalar
        sub       = x_softmax[j_start : j_start + 64]  # [64]
        x_mean    = x_mean + x_local * (tl.sum(sub, axis=0) / 4096.0)
        y_mean    = y_mean + y_local * (tl.sum(sub, axis=0) / 4096.0)

    # ── Store outputs ────────────────────────────────────────────────────────
    out_base = b * 278528 + h * 4096          # 17 * 64 * 64 = 69632
    tl.store(out_ptr      + out_base + j, x_invlogit.to(x_vals.dtype))

    slot_offsets = slot * 2          # x slots: 0..63 → 0..127
    tl.store(sum_out_ptr  + pid * 128 + slot_offsets,
             x_mean.to(x_vals.dtype))
    tl.store(sum_out_ptr  + pid * 128 + slot_offsets + 64,
             y_mean.to(x_vals.dtype))


# ── Batch-agnostic wrapper ─────────────────────────────────────────────────

@torch.fx.wrap
def fused_softmax_weightedmean(in_0, in_1, in_2):
    """
    in_0 : [1, 1, 1, 64]   (linspace_x)
    in_1 : [1, 1, 64, 1]   (linspace_y)
    in_2 : [BATCH, 17, 4096]  (primary feature maps)
    returns: (logits[Fortrans,17,64,64],  sum_y_combined[B,17,128])
    """
    BATCH = in_2.shape[0]
    out   = torch.empty((BATCH, 17, 64, 64), dtype=in_2.dtype, device=in_2.device)
    sum_out = torch.empty((BATCH * 17, 128), dtype=in_2.dtype, device=in_2.device)

    fused_softmax_weightedmean_kernel[(BATCH * 17,)](
        logits_ptr=in_2,
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        sum_out_ptr=sum_out,
        BATCH=BATCH,
        num_warps=8,
    )

    return out, sum_out.reshape(BATCH, 17, 128)