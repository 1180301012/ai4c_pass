"""
Shared fused add+transpose+layer_norm Triton kernel.
Reads conv_out and x (stride-256 for transposed layout),
fuses add + layer_norm, writes transposed output.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_transpose_ln(
    conv_ptr,   # [1, C, H*W] – conv2d output  (contiguous: stride [C*HW, HW, 1])
    x_ptr,      # [1, C, H*W] – residual        (same layout)
    w_ptr,      # [C] – LN weight (contiguous)
    b_ptr,      # [C] – LN bias   (contiguous)
    trans_ptr,  # [1, HW, C] – pre-LN transposed output  (may be non-contiguous)
    y_ptr,      # [1, HW, C] – LN output               (contiguous)
    N,          # HW = number of spatial positions (256)
    C,          # channels (768 or 1024)
    sx1, sx2,   # conv_ptr strides [dim-1=spatial, dim-2=channel]
    sy1, sy2,   # trans_ptr strides (preserved from non-contiguous layout)
    eps,
    BLOCK_C: tl.constexpr,
):
    row  = tl.program_id(0)    # one program per spatial position
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    base = row * sx1   # base for this row (stride-sx1 apart in channel dim)

    # ── Load conv output and residual (stride sx2 = HW between channels) ──────
    a = tl.load(conv_ptr + base + cols * sx2, mask=mask, other=0.0)
    b = tl.load(x_ptr    + base + cols * sx2, mask=mask, other=0.0)
    add_val = a + b   # the add result; kept in registers

    # ── Write pre-LN transposed output (same non-contiguous layout as tmp_7) ─
    tl.store(trans_ptr + row * sy1 + cols * sy2, add_val.to(trans_ptr.dtype.element_ty), mask=mask)

    # ── Layer-norm over this spatial position (768 channels) ──────────────────
    xf      = add_val.to(tl.float32)
    mean    = tl.sum(xf, axis=0) / C
    diff    = tl.where(mask, xf - mean, 0.0)
    var     = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.rsqrt(var + eps)

    w   = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y   = diff * inv_std * w + bias

    # ── Write LN output (contiguous row-major [N, C]) ─────────────────────────
    tl.store(y_ptr + row * C + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def launch_fused_add_ln(x, conv_out, weight, bias):
    """
    x, conv_out : [1, C, H, W] contiguous
    returns     : (trans_out [1,H*W,C] non-contig, y_ln [1,H*W,C] contig)
    """
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    N  = HW                         # batch=1

    trans = torch.empty((1, HW, C), dtype=x.dtype, device=x.device)
    y     = torch.empty((1, HW, C), dtype=x.dtype, device=x.device)

    sx1 = x.stride(1)   # spatial stride  (= H*W = 256 for [1,C,16,16])
    sx2 = x.stride(2)   # channel stride  (= 1 for contiguous)
    # trans is non-contiguous: same strides as tmp_7 = [N*C*HW, 1, HW]
    sy1 = trans.stride(-2)   # = sx1 = 256
    sy2 = trans.stride(-1)   # = sx2 = 1

    BLOCK_C = 1024 if C <= 1024 else 2048

    _fused_add_transpose_ln[(N,)](
        conv_out, x, weight, bias,
        trans, y,
        N, C,
        sx1, sx2,
        sy1, sy2,
        1e-5,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    return trans, y