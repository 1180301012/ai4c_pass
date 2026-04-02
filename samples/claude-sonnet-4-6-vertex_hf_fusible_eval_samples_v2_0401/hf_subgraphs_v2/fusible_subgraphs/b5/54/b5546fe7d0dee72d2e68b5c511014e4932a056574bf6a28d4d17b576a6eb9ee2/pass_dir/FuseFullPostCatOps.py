"""
Full-fusion pass: fuses  transpose(-1,-2) + mul + pad + add + transpose(1,2)
into a single Triton kernel.

This saves three intermediate tensor materializations:
  tmp6 = in6 * transposed_tmp4   [HW, K] - skipped
  tmp7 = pad(tmp6)               [HW+1, K] - skipped
  tmp9 = scaled_in4 + tmp7       [HW+1, K] - output directly in transposed layout

Uses a LIST [0,0,1,0,0,0] for the pad argument in case the serialised FX
graph stores it as a list rather than a tuple.

Pattern:
    tmp5  = tmp4.transpose(-1, -2)
    tmp6  = in6 * tmp5
    tmp7  = F.pad(tmp6, [0,0,1,0,0,0], 'constant', None)
    tmp9  = scaled_in4 + tmp7
    tmp10 = tmp9.transpose(1, 2)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – list notation for pad args
# ---------------------------------------------------------------------------

def pattern(tmp4, in6, scaled_in4):
    """
    tmp4:       [B, H, K, HW]
    in6:        [B, H, HW, K]
    scaled_in4: [B, H, HW+1, K]  (= scale * in_4, pre-computed)
    """
    tmp5  = tmp4.transpose(-1, -2)
    tmp6  = in6 * tmp5
    tmp7  = torch.nn.functional.pad(tmp6, [0, 0, 1, 0, 0, 0], 'constant', None)
    tmp9  = scaled_in4 + tmp7
    tmp10 = tmp9.transpose(1, 2)
    return tmp10


def replacement_args(tmp4, in6, scaled_in4):
    return (tmp4, in6, scaled_in4)


# ---------------------------------------------------------------------------
# Triton kernel: full fusion
# out[b, i, h, k] = scaled_in4[b,h,i,k]
#                 + (i>0 ? in6[b,h,i-1,k] * tmp4[b,h,k,i-1] : 0)
# ---------------------------------------------------------------------------

@triton.jit
def _full_fused_kernel(
    tmp4_ptr, in6_ptr, sin4_ptr, out_ptr,
    H, HW1, K,
    BLOCK_K: tl.constexpr,
):
    """
    Grid = (H * HW1,).  One program per (h, i) pair.

    tmp4  [1, H, K, HW]: element [h, k, hw]  at  h*K*HW + k*HW + hw
    in6   [1, H, HW, K]: element [h, hw, k]  at  h*HW*K + hw*K + k
    sin4  [1, H, HW+1,K]: element [h, i, k]  at  h*HW1*K + i*K + k
    out   [1, HW+1,H, K]: element [i, h, k]  at  i*H*K  + h*K  + k
    """
    pid = tl.program_id(0)
    h   = pid % H
    i   = pid // H

    k    = tl.arange(0, BLOCK_K)
    mask = k < K

    HW  = HW1 - 1          # original spatial size (before padding)

    # Load sin4[h, i, k]
    s_off  = h * HW1 * K + i * K + k
    s_val  = tl.load(sin4_ptr + s_off, mask=mask, other=0.0)

    # For i > 0: load in6[h, i-1, k] and tmp4[h, k, i-1]
    i_gt0   = i > 0
    im1     = tl.where(i_gt0, i - 1, 0)            # clamped

    # in6[h, im1, k]: stride_k=1 → coalesced
    in6_off  = h * HW * K + im1 * K + k
    in6_val  = tl.load(in6_ptr + in6_off, mask=(mask & i_gt0), other=0.0)

    # tmp4[h, k, im1]: stride over k (tmp4 stride_k = HW, non-coalesced)
    tmp4_off = h * K * HW + k * HW + im1
    tmp4_val = tl.load(tmp4_ptr + tmp4_off, mask=(mask & i_gt0), other=0.0)

    mul_val = in6_val * tmp4_val
    result  = s_val + tl.where(i_gt0, mul_val, 0.0)

    # Store to out[i, h, k]: stride_k=1 → coalesced
    out_off = i * H * K + h * K + k
    tl.store(out_ptr + out_off, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_full_postcat(tmp4, in6, scaled_in4):
    """
    tmp4:       [B, H, K, HW]
    in6:        [B, H, HW, K]
    scaled_in4: [B, H, HW+1, K]
    Returns:    contiguous [B, HW+1, H, K]
    """
    B, H, K, HW = tmp4.shape
    HW1 = HW + 1
    out  = torch.empty(B, HW1, H, K, dtype=tmp4.dtype, device=tmp4.device)
    grid = (H * HW1,)
    if K <= 32:
        _full_fused_kernel[grid](tmp4, in6, scaled_in4, out, H, HW1, K, BLOCK_K=32)
    else:
        _full_fused_kernel[grid](tmp4, in6, scaled_in4, out, H, HW1, K, BLOCK_K=64)
    return out


def replacement_func():
    return fused_full_postcat