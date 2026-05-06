import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused softmax-weighted sum
#
# in_1 : [B, 2, 1, C]  (logits for 2 heads)
# in_0 : [B, 2, C, H, W]  (feature maps)
# out  : [B, C, H, W]
#
# For each (b,c) program:
#   softmax over in_1[b, :, 0, c] → w0, w1
#   out[b,c,:] = w0 * in_0[b,0,c,:] + w1 * in_0[b,1,c,:]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _fused_attn_kernel(
    in0_ptr,   # [B, 2, C, H, W]  contiguous
    in1_ptr,   # [B, 2, 1, C]     contiguous
    out_ptr,   # [B, C, H, W]     contiguous
    B, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # One program per (b, c) pair
    bc = tl.program_id(0)
    b  = bc // C
    c  = bc  % C

    # ---- softmax over the 2 head vectors --------------------------------
    # in_1 strides for [B, 2, 1, C]:  stride_b=2*C, stride_h=C, stride_c=1
    idx0 = b * (2 * C) + c          # head-0 element for (b,c)
    idx1 = b * (2 * C) + C + c      # head-1 element for (b,c)

    v0 = tl.load(in1_ptr + idx0).to(tl.float32)
    v1 = tl.load(in1_ptr + idx1).to(tl.float32)

    m  = tl.maximum(v0, v1)
    e0 = tl.exp(v0 - m)
    e1 = tl.exp(v1 - m)
    den = e0 + e1
    w0 = e0 / den
    w1 = e1 / den

    # ---- weighted sum over spatial positions ----------------------------
    base = b * (2 * C * HW) + c * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for hw in range(0, HW, BLOCK_HW):
        off  = hw + tl.arange(0, BLOCK_HW)
        mask = off < HW

        # bc*HW + h*C*HW + hw == in_0[b, h, c, hw] offset  ✓
        in0_0 = tl.load(in0_ptr + bc * HW + off, mask=mask, other=0.0).to(tl.float32)
        in0_1 = tl.load(in0_ptr + (bc + C * HW) + off, mask=mask, other=0.0).to(tl.float32)

        acc = acc + w0 * in0_0 + w1 * in0_1

    out_idx = bc * HW + tl.arange(0, BLOCK_HW)
    tl.store(out_ptr + out_idx, acc, mask=out_idx < B * C * HW)


@torch.fx.wrap
def fused_attention_view_mul_sum(in_0, tmp_0):
    B  = in_0.shape[0]
    C  = in_0.shape[1]
    HW = in_0.shape[2] * in_0.shape[3]
    out = torch.empty((B, C, in_0.shape[2], in_0.shape[3]),
                      dtype=in_0.dtype, device=in_0.device)
    _fused_attn_kernel[(B * C,)](in_0, tmp_0, out, B, C, HW)
    return out


def replacement_func():
    return fused_attention_view_mul_sum