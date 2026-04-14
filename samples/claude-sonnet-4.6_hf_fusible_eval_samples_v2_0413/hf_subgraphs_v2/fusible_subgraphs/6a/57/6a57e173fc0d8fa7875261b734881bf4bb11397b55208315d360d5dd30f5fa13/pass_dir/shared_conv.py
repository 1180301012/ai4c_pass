"""
Shared Triton kernel: fused permute(0,2,1,3) + contiguous + view.

Input layout : [B, G, H, W]  (result of conv2d + iadd already done)
Output layout: [B, H, G*W]   (permute swaps dims 1<->2, then flatten last two)
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: strided copy  in[b,g,h,w] → out[b,h,g*W+w]
# Grid = (B, H)  – one program handles the G*W elements for one (b, h) pair
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _permute_view_kernel(
    in_ptr,           # [B, G, H, W]
    out_ptr,          # [B, H, G*W]
    B, G, H, W,
    in_stride_b, in_stride_g, in_stride_h, in_stride_w,
    out_stride_b, out_stride_h,
    BLOCK_W: tl.constexpr,        # >= W, power-of-2
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_b
    h = pid_h

    w_arange = tl.arange(0, BLOCK_W)
    w_mask   = w_arange < W

    for g in range(G):
        in_offs = (b * in_stride_b + g * in_stride_g +
                   h * in_stride_h + w_arange * in_stride_w)
        vals = tl.load(in_ptr + in_offs, mask=w_mask)

        out_offs = b * out_stride_b + h * out_stride_h + g * W + w_arange
        tl.store(out_ptr + out_offs, vals, mask=w_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_permute_contiguous_view(in_3):
    """
    Fused: permute(0,2,1,3) + contiguous + view(B, H, G*W)

    in_3 : [B, G, H, W]  – any dtype
    out  : [B, H, G*W]
    """
    B, G, H, W = in_3.shape
    GW  = G * W
    out = torch.empty((B, H, GW), dtype=in_3.dtype, device=in_3.device)

    in_stride_b, in_stride_g, in_stride_h, in_stride_w = in_3.stride()
    out_stride_b = H * GW
    out_stride_h = GW

    BLOCK_W = triton.next_power_of_2(W)

    grid = (B, H)
    _permute_view_kernel[grid](
        in_3, out,
        B, G, H, W,
        in_stride_b, in_stride_g, in_stride_h, in_stride_w,
        out_stride_b, out_stride_h,
        BLOCK_W=BLOCK_W,
    )
    return out