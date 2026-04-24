import torch
import torch.fx
import inspect
import operator
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match getitem(x, (Ellipsis, slice(None, 64, None)))
# This matches the slice that produces tmp_4 = tmp_3[...,:64].
# x = tmp_3, shape [B, H, W, 2J=128]
# ---------------------------------------------------------------------------
def pattern(x):
    return x[(Ellipsis, slice(None, 64, None))]


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: copy first J elements from [B,H,W,2J] tensor to [B,H,W,J]
# Grid: (B * H * W,)
# ---------------------------------------------------------------------------
@triton.jit
def _slice_kernel(
    in_ptr, out_ptr,
    B, H, W, J,
    in_s0, in_s1, in_s2,
    out_s0, out_s1, out_s2,
    BLOCK_J: tl.constexpr,
):
    row = tl.program_id(0)
    b  = row // (H * W)
    hw = row  % (H * W)
    h  = hw   // W
    w  = hw   %  W

    j_offs = tl.arange(0, BLOCK_J)

    in_base  = b * in_s0 + h * in_s1 + w * in_s2
    out_base = b * out_s0 + h * out_s1 + w * BLOCK_J

    # Copy first J elements: in[b,h,w,0:J]
    val = tl.load(in_ptr + in_base + j_offs)
    tl.store(out_ptr + out_base + j_offs, val)


# ---------------------------------------------------------------------------
# Wrapper — returns a proper view-equivalent slice of the first J elements
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_slice(x):
    """
    x   : [B, H, W, 2J]  (softmax output, 2J = 128)
    returns [B, H, W, J]  (equivalent to x[...,:J])
    """
    B, H, W, J2 = x.shape
    J = J2 // 2

    out = torch.empty((B, H, W, J), dtype=x.dtype, device=x.device)

    _slice_kernel[(B * H * W,)](
        x, out,
        B, H, W, J,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_J=64,
        num_warps=4,
    )

    return out


def replacement_func():
    return triton_slice