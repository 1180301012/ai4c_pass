import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: softmax(in_1, dim=1)  *  in_0  →  sum over dim=1
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel — 2D grid (C, ceil(H/BLOCK_HW))
#
#   dim-0  = channel index c  →  fixed channel per program
#   dim-1  = spatial block    →  BLOCK_HW contiguous spatial positions
#
# For in_0 [B,2,C,H,W]: flat[k,c,h,w] = k*C_HW + c*HW + h*W + w
#   where HW = H*W,  HW_stride = W
# For in_1 [B,2,C,1,1]: flat[k,c]     = k*C + c
# For out  [B,C,H,W]:   flat[c,h,w]   = c*HW + h*W + w
#
# Passes H and W separately so the wrapper can produce correct output shape.
# ---------------------------------------------------------------------------

@triton.jit
def fused_softmax_weighted_sum_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    C_HW,               # C * H * W  (total elements per k-slice)
    HW,                 # H * W
    W,                  # W  (spatial width, used for flat index in out)
    C,                  # channel count
    in1_row_offset,     # C  (k=1 base offset in in_1)
    BLOCK_HW: tl.constexpr,
):
    c        = tl.program_id(0)
    hw_blk   = tl.program_id(1)
    hw_start = hw_blk * BLOCK_HW

    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW

    # ---- two scalar logits for channel c ----
    log0 = tl.load(in1_ptr + c).to(tl.float32)
    log1 = tl.load(in1_ptr + in1_row_offset + c).to(tl.float32)

    # ---- numerically-stable softmax (K=2) ----
    max_val = tl.maximum(log0, log1)
    s0 = tl.exp(log0 - max_val) / (tl.exp(log0 - max_val) + tl.exp(log1 - max_val))
    s1 = tl.exp(log1 - max_val) / (tl.exp(log0 - max_val) + tl.exp(log1 - max_val))

    # ---- perfectly coalesced vector loads (contiguous hw within channel c) ----
    in0_base = c * HW
    v0 = tl.load(in0_ptr + in0_base + hw_offs,          mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in0_ptr + C_HW + in0_base + hw_offs,   mask=mask, other=0.0).to(tl.float32)

    # ---- weighted sum, cast to input dtype, store ----
    result = (v0 * s0 + v1 * s1).to(v0.dtype)
    tl.store(out_ptr + c * HW + hw_offs, result, mask=mask)


_BLOCK_HW = 256
_NWARPS   = 8   # 256 threads / 8 warps = 1 element per thread


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1):
    """
    in_0 : [B, 2, C, H, W]
    in_1 : [B, 2, C, 1, 1]
    out  : [B, C, H, W]   ← preserve original H and W
    """
    shape = in_0.shape
    B  = shape[0]
    C  = shape[2]    # channel count (dim-2 of input)
    H  = shape[3]    # spatial height
    W  = shape[4]    # spatial width
    HW       = H * W
    C_HW     = C * HW
    n_hw_blks = (HW + _BLOCK_HW - 1) // _BLOCK_HW

    # Output preserves original (H, W) dimensions — not HW as a flat dimension
    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    fused_softmax_weighted_sum_kernel[(C, n_hw_blks)](
        in_0, in_1, out,
        C_HW, HW, W, C, C,    # in1_row_offset = C for k=1
        BLOCK_HW=_BLOCK_HW,
        num_warps=_NWARPS,
    )
    return out


def replacement_func():
    return fused_softmax_weighted_sum