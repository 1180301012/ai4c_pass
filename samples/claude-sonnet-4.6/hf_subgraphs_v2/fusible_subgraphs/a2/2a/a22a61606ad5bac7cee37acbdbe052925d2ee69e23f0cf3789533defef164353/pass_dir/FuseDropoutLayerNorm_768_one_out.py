"""
Pass: FuseDropoutLayerNorm_768_one_out
Fuses dropout (no-op, training=False) + layer_norm(hidden=768) into a single
Triton kernel.  Returns ONLY the layer-norm output (the dropout intermediate
is not exposed in the model's return, so it is consumed inside the pattern).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel (inlined)
# ---------------------------------------------------------------------------
@triton.jit
def _ln_fwd_768_one(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_orig = tl.load(X_ptr + row * stride + offsets, mask=mask, other=0.0)
    x = x_orig.to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    xn = xc * rstd

    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = xn * w + b
    tl.store(Y_ptr + row * stride + offsets, y.to(x_orig.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(x, ln_weight, ln_bias):
    drop = torch.nn.functional.dropout(x, 0.1, False, False)
    ln   = torch.nn.functional.layer_norm(drop, (768,), ln_weight, ln_bias, 1e-12)
    return ln


# ---------------------------------------------------------------------------
# Triton-based wrapper (single output)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_dropout_ln_768_one_out(x, ln_weight, ln_bias):
    N = 768
    M = x.numel() // N
    out = torch.empty_like(x)

    _ln_fwd_768_one[(M,)](
        x, ln_weight, ln_bias, out,
        N, N, 1e-12,
        BLOCK_SIZE=1024,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------
def replacement_args(x, ln_weight, ln_bias):
    return (x, ln_weight, ln_bias)


def replacement_func():
    return triton_dropout_ln_768_one_out