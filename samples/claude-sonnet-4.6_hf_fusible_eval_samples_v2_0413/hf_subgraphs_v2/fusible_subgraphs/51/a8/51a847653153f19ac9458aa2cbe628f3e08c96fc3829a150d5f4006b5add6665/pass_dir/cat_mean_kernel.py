import torch
import triton
import triton.language as tl


def _npow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


# Single kernel: fused cat of in0 and in1 into out
@triton.jit
def _ck_cat(in0_ptr, in1_ptr, out_ptr, C, C2, HW, BLOCK_HW: tl.constexpr):
    """
    Grid: (B * C2,)  —  one program per (batch, output-channel).
    First C channels come from in0, next C from in1.
    No branch: selects source using arithmetic.
    """
    pid   = tl.program_id(0)
    b_idx = pid // C2
    c_out = pid % C2
    # Source channel: for c_out < C use in0[b, c_out], else in1[b, c_out-C]
    is_in1    = (c_out >= C)
    is_in1_i  = is_in1.to(tl.int32)
    c_src     = c_out - C * is_in1_i          # safe: >=0 in both branches
    in0_base  = (b_idx * C + c_out) * HW      # used when ~is_in1
    in1_base  = (b_idx * C + c_src)  * HW     # used when  is_in1
    dst_base  = (b_idx * C2 + c_out) * HW
    offs = tl.arange(0, BLOCK_HW)
    hw   = offs < HW
    v0 = tl.load(in0_ptr + in0_base + offs, mask=hw & ~is_in1, other=0.0)
    v1 = tl.load(in1_ptr + in1_base + offs, mask=hw &  is_in1, other=0.0)
    tl.store(out_ptr + dst_base + offs, v0 + v1, mask=hw)


# @torch.fx.wrap hides Triton; returns a SINGLE tensor (avoids multi-output)
@torch.fx.wrap
def cat_fused(in_0, in_1):
    B  = in_0.shape[0]; C  = in_0.shape[1]
    H  = in_0.shape[2]; W  = in_0.shape[3]
    HW = H * W;         C2 = C + C
    BLK = _npow2(HW);   nw = max(1, min(16, BLK // 64))
    out = torch.empty((B, C2, H, W), dtype=in_0.dtype, device=in_0.device)
    _ck_cat[(B * C2,)](in_0, in_1, out, C, C2, HW, BLOCK_HW=BLK, num_warps=nw)
    return out