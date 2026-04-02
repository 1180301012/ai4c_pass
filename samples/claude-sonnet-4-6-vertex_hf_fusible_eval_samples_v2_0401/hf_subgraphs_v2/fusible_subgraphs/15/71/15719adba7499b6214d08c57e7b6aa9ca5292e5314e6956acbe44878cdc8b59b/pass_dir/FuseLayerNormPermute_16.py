import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: layer norm for C=16, N=256 patches
# tmp_7 is non-contiguous: shape [1,256,16], strides [4096,1,256]
# ---------------------------------------------------------------------------

@triton.jit
def _layer_norm_kernel_16(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    x_stride_row,    # stride along patch dim  (runtime: x.stride(1))
    x_stride_col,    # stride along channel dim (runtime: x.stride(2))
    eps: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    One program per patch. Handles non-contiguous input via runtime strides.
    Output is written contiguously (stride = 1 for channel dim).
    """
    pid  = tl.program_id(0)
    C    = 16
    chan = tl.arange(0, 16)

    # Load input with actual (potentially non-contiguous) strides
    in_offs = pid * x_stride_row + chan * x_stride_col
    x       = tl.load(x_ptr + in_offs)
    x_f32   = x.to(tl.float32)

    # Layer norm
    mean    = tl.sum(x_f32, 0) / C
    diff    = x_f32 - mean
    var     = tl.sum(diff * diff, 0) / C
    inv_std = tl.rsqrt(var + eps)
    x_norm  = diff * inv_std

    # Affine transform
    w = tl.load(w_ptr + chan).to(tl.float32)
    b = tl.load(b_ptr + chan).to(tl.float32)
    y = x_norm * w + b

    if IS_BF16:
        y_out = y.to(tl.bfloat16)
    else:
        y_out = y.to(tl.float16)

    # Write contiguous output
    tl.store(out_ptr + pid * C + chan, y_out)


@torch.fx.wrap
def triton_layer_norm_16(x, weight, bias):
    """Replace layer_norm(16) + dropout(p=0). Handles non-contiguous x."""
    # Always create a fresh contiguous output tensor
    out     = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    IS_BF16 = (x.dtype == torch.bfloat16)
    _layer_norm_kernel_16[(256,)](
        x_ptr=x,
        w_ptr=weight,
        b_ptr=bias,
        out_ptr=out,
        x_stride_row=x.stride(1),
        x_stride_col=x.stride(2),
        eps=1e-5,
        IS_BF16=IS_BF16,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement glue
# ---------------------------------------------------------------------------

def pattern(tmp_7, in_2, in_1):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1)


def replacement_func():
    return triton_layer_norm_16