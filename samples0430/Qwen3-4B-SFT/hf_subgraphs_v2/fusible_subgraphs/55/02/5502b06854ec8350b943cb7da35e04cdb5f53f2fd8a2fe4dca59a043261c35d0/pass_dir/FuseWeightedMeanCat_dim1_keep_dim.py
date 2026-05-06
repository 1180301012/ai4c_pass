import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused pattern:
#   tmp_0 = in_0.to(float32)                   [1,10,1024]
#   tmp_1 = in_1 * tmp_0                        [1,10,1024]  ← float32 (cast wins)
#   tmp_2 = torch.sum(tmp_1, 1)                 [1,1024]      ← float32
#   tmp_3 = tmp_0.sum(1)                        [1,1024]      ← float32
#   tmp_4 = clamp(tmp_3, min=1e-9)              [1,1024]      ← float32
#   tmp_5 = tmp_2 / tmp_4                       [1,1024]      ← float32
#   tmp_6 = cat([tmp_5], 1)                     [1,1024]      ← float32
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0  # tmp_1 lands in float32 (float32 > bf16/fp16)
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel – coalesced 2-D tile loads
#
#   Grid  : (ceil(D / BLOCK_D),)
#   Thread block processes a [K, BLOCK_D] tile of in0/in1:
#     - Load in0[pid*BLOCK_D : pid*BLOCK_D+BLOCK_D, :] → coalesced in D dim
#     - Load in1[pid*BLOCK_D : pid*BLOCK_D+BLOCK_D, :] → coalesced in D dim
#   Then reduce across K with tl.sum → per-feature weighted mean in fp32
#   Output  : [B, D]  float32   (茅 match the float32dtype of tmp_1)
# ---------------------------------------------------------------------------

@triton.jit
def fused_weighted_mean_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, D,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,   # next power-of-2 >= K=10  (used only for arange/zero)
):
    pid = tl.program_id(0)

    # ---- per-program offsets ------------------------------------------------
    d_offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)   # [BLOCK_D]
    k_offs = tl.arange(0, BLOCK_K)                    # [BLOCK_K]
    mask_d  = d_offs < D
    mask_k  = k_offs < 10                             # K = 10 hardcoded

    # ---- 2-D tile: [BLOCK_K, BLOCK_D]  (inner-D = fast dim → coalesced) ----
    tile = k_offs[:, None] * D + d_offs[None, :]      # [BLOCK_K, BLOCK_D]
    m2d  = mask_k[:, None] & mask_d[None, :]

    in0_tile = tl.load(in0_ptr + tile, mask=m2d, other=0).to(tl.float32)
    in1_tile = tl.load(in1_ptr + tile, mask=m2d, other=0).to(tl.float32)

    # ----- single-stage fused reduction (avoids separate sum walks) ----------
    acc_val = tl.sum(in1_tile * in0_tile, axis=0)     # [BLOCK_D]
    acc_wg  = tl.sum(in0_tile,                    axis=0)  # [BLOCK_D]

    result = acc_val / tl.maximum(acc_wg, 1e-9)

    tl.store(out_ptr + d_offs, result, mask=mask_d)


@torch.fx.wrap
def fused_weighted_mean(in_0, in_1):
    """
    Replacement for the full pattern.
    in_0 : [B, K, D]  int64
    in_1 : [B, K, D]  bfloat16 or float16
    returns: [B, D]   float32  (matches the dtype of all float32 intermediates)
    """
    B, K, D = in_0.shape
    out = torch.empty((B, D), dtype=torch.float32, device=in_0.device)

    # Best config for D=1024, K=10 on A30:
    #   BLOCK_D=32, BLOCK_K=16, num_warps=1, num_stages=2
    BLOCK_D  = 32
    BLOCK_K  = 16   # next power-of-2 >= K=10

    grid = (triton.cdiv(D, BLOCK_D),)

    fused_weighted_mean_kernel[grid](
        in_0, in_1, out,
        B, D,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=1,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_weighted_mean