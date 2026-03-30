import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match adaptive_avg_pool2d(in_0, (32, 24)) followed by cat([pooled, in_1], dim=1).
    in_0: [B, C0, 64, 48]  ->  pooled: [B, C0, 32, 24]
    in_1: [B, C1, 32, 24]
    output: [B, C0+C1, 32, 24]
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_avgpool_cat_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C0, C1,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single fused kernel with 3D grid: (b, c, spatial_block).
    - No per-element division by C_out (decoded from grid dims).
    - OW/OH/IH/IW are compile-time constants → mul-shift optimisation.
    - c is a scalar per block → branch elimination for pool vs copy.
    For each output element at (b, c, oh, ow):
      - c < C0:  output = avg_pool_2x2(in_0[b, c, oh*2:oh*2+2, ow*2:ow*2+2])
      - c >= C0: output = in_1[b, c-C0, oh, ow]
    """
    b      = tl.program_id(0)   # batch index
    c      = tl.program_id(1)   # channel index 0 … C0+C1-1
    sp_pid = tl.program_id(2)   # spatial tile

    C_out   = C0 + C1
    is_pool = c < C0             # scalar per block → dead-code-elim of unused branch

    sp   = sp_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sp < OH * OW          # OH*OW = 32*24 = 768

    # Output flat offset: [B, C_out, OH, OW]  — sp = oh*OW+ow is the flat spatial index
    out_off = b * (C_out * OH * OW) + c * (OH * OW) + sp

    if is_pool:
        # ── Avg-pool path: in_0 [B, C0, IH, IW], 2×2 → [B, C0, OH, OW] ───
        # Spatial decode only needed for pool path; constexpr OW → mul-shift trick
        oh = sp // OW
        ow = sp % OW
        h0 = oh * 2          # IH = 2*OH
        w0 = ow * 2          # IW = 2*OW
        in0_off = (b * (C0 * IH * IW)
                   + c  * (IH * IW)
                   + h0 * IW
                   + w0)

        v00 = tl.load(in0_ptr + in0_off,          mask=mask, other=0.0)
        v01 = tl.load(in0_ptr + in0_off + 1,      mask=mask, other=0.0)
        v10 = tl.load(in0_ptr + in0_off + IW,     mask=mask, other=0.0)
        v11 = tl.load(in0_ptr + in0_off + IW + 1, mask=mask, other=0.0)
        pool_val = (v00 + v01 + v10 + v11) * 0.25
        tl.store(out_ptr + out_off, pool_val, mask=mask)
    else:
        # ── Copy path: in_1 [B, C1, OH, OW] → out channels C0..C0+C1-1 ────
        c1 = c - C0
        in1_off = b * (C1 * OH * OW) + c1 * (OH * OW) + sp
        copy_val = tl.load(in1_ptr + in1_off, mask=mask, other=0.0)
        tl.store(out_ptr + out_off, copy_val, mask=mask)


@torch.fx.wrap
def fused_avgpool_cat(in_0, in_1):
    B,  C0, IH, IW = in_0.shape
    _,  C1, OH, OW = in_1.shape
    C_out = C0 + C1

    out = torch.empty((B, C_out, OH, OW), dtype=in_0.dtype, device=in_0.device)

    # Fixed BLOCK_SIZE = 256: 100% spatial utilization (768/256 = 3 exact tiles),
    # more SM-level parallelism for large batches.
    BLOCK_SIZE = 256
    grid = (B, C_out, triton.cdiv(OH * OW, BLOCK_SIZE))

    fused_avgpool_cat_kernel[grid](
        in_0, in_1, out,
        B, C0, C1,
        OH=OH, OW=OW, IH=IH, IW=IW,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_avgpool_cat