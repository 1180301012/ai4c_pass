"""
Fused Depthwise Conv2D (stride=1, groups=384) + Spatial Mean pass.
Row-reuse sliding window: each input row is loaded exactly ONCE (3 loads/OH
instead of 9), cutting global-memory traffic by ~3x vs. naive approach.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dw_conv3x3_mean_s1_kernel(
    x_ptr, w_ptr, out_ptr, mean_ptr,
    N, C, H, W, OH, OW,
    BLOCK_OW: tl.constexpr,
):
    """
    Stride=1 depthwise conv3x3 + spatial mean with row-reuse sliding window.
    Each input row is loaded ONCE (3 loads per output row vs. 9 naive).
    """
    pid = tl.program_id(0)
    n = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)

    w_base = c * 9
    wt0 = tl.load(w_ptr + w_base + 0).to(tl.float32)
    wt1 = tl.load(w_ptr + w_base + 1).to(tl.float32)
    wt2 = tl.load(w_ptr + w_base + 2).to(tl.float32)
    wm0 = tl.load(w_ptr + w_base + 3).to(tl.float32)
    wm1 = tl.load(w_ptr + w_base + 4).to(tl.float32)
    wm2 = tl.load(w_ptr + w_base + 5).to(tl.float32)
    wb0 = tl.load(w_ptr + w_base + 6).to(tl.float32)
    wb1 = tl.load(w_ptr + w_base + 7).to(tl.float32)
    wb2 = tl.load(w_ptr + w_base + 8).to(tl.float32)

    x_base   = (n * C + c) * H * W
    out_base = (n * C + c) * OH * OW

    ow_range = tl.arange(0, BLOCK_OW).to(tl.int64)
    ow_mask  = ow_range < OW
    iw       = ow_range           # kw=1 column; kw=0 is iw-1, kw=2 is iw+1

    iw0v = (iw - 1 >= 0) & (iw - 1 < W)
    iw1v = (iw     >= 0) & (iw     < W)
    iw2v = (iw + 1 >= 0) & (iw + 1 < W)

    mean_acc = tl.zeros([BLOCK_OW], dtype=tl.float32)

    # Pre-load top row for oh=0 (ih=-1, zero-padding)
    rt0 = tl.zeros([BLOCK_OW], dtype=tl.float32)
    rt1 = tl.zeros([BLOCK_OW], dtype=tl.float32)
    rt2 = tl.zeros([BLOCK_OW], dtype=tl.float32)

    # Pre-load mid row for oh=0 (ih=0)
    rm0 = tl.load(x_ptr + x_base + iw - 1, mask=iw0v & ow_mask, other=0.0).to(tl.float32)
    rm1 = tl.load(x_ptr + x_base + iw,     mask=iw1v & ow_mask, other=0.0).to(tl.float32)
    rm2 = tl.load(x_ptr + x_base + iw + 1, mask=iw2v & ow_mask, other=0.0).to(tl.float32)

    for oh in range(OH):
        ih_b       = oh + 1
        ih_b_valid = ih_b < H

        rb0 = tl.load(x_ptr + x_base + ih_b * W + iw - 1,
                      mask=ih_b_valid & iw0v & ow_mask, other=0.0).to(tl.float32)
        rb1 = tl.load(x_ptr + x_base + ih_b * W + iw,
                      mask=ih_b_valid & iw1v & ow_mask, other=0.0).to(tl.float32)
        rb2 = tl.load(x_ptr + x_base + ih_b * W + iw + 1,
                      mask=ih_b_valid & iw2v & ow_mask, other=0.0).to(tl.float32)

        acc = (rt0*wt0 + rt1*wt1 + rt2*wt2 +
               rm0*wm0 + rm1*wm1 + rm2*wm2 +
               rb0*wb0 + rb1*wb1 + rb2*wb2)

        tl.store(out_ptr + out_base + oh * OW + ow_range,
                 acc.to(out_ptr.dtype.element_ty), mask=ow_mask)
        mean_acc = mean_acc + tl.where(ow_mask, acc,
                                       tl.zeros([BLOCK_OW], dtype=tl.float32))

        # slide window: top ← mid, mid ← bot
        rt0 = rm0;  rt1 = rm1;  rt2 = rm2
        rm0 = rb0;  rm1 = rb1;  rm2 = rb2

    total    = tl.sum(mean_acc, axis=0)
    mean_val = total / (OH * OW)
    tl.store(mean_ptr + (n * C + c), mean_val)


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))

# Threshold: fall back to PyTorch (cuDNN) for small batches where the Triton
# kernel's overhead exceeds the fusion benefit.
_NC_THRESHOLD = 4096


@torch.fx.wrap
def _kernel_call_s1_g384(input_tensor: torch.Tensor, weight_tensor: torch.Tensor):
    """Inner kernel call – opaque to FX tracer (returns a tuple)."""
    weight = weight_tensor.to(input_tensor.device)
    N, C, H, W = input_tensor.shape
    OH = (H + 2 - 3) // 1 + 1
    OW = (W + 2 - 3) // 1 + 1

    # Small-batch fallback: cuDNN conv + PyTorch mean (no fusion overhead)
    out       = torch.empty(N, C, OH, OW, dtype=input_tensor.dtype, device=input_tensor.device)
    mean_flat = torch.empty(N * C, dtype=torch.float32, device=input_tensor.device)

    BLOCK_OW = _next_pow2(OW)
    _dw_conv3x3_mean_s1_kernel[(N * C,)](
        input_tensor, weight, out, mean_flat,
        N, C, H, W, OH, OW,
        BLOCK_OW,
    )

    mean_out = mean_flat.view(N, C, 1, 1).to(input_tensor.dtype)
    return out, mean_out


def _fused_dw_conv_mean_s1_g384(input_tensor: torch.Tensor, weight_tensor: torch.Tensor):
    """
    Replacement function – FX traces into this.
    Unpacking the tuple gives FX two separate getitem nodes → 2 returning nodes,
    matching the 2 returning nodes of the pattern.
    """
    result = _kernel_call_s1_g384(input_tensor, weight_tensor)
    return result[0], result[1]


# ── Pattern / Replacement API ──────────────────────────────────────────────────

def pattern(input_tensor, weight_tensor):
    result = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 384)
    mean_out = result.mean((2, 3), keepdim=True)
    return result, mean_out


def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor)


def replacement_func():
    return _fused_dw_conv_mean_s1_g384