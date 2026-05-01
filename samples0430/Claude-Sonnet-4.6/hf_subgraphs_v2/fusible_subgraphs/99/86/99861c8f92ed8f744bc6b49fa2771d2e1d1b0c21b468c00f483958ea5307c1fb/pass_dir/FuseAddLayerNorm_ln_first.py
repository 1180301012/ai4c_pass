import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_layernorm_lnf_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    sum_ptr, out_ptr,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    # Load inputs
    x = tl.load(x_ptr + row * N + cols)
    y = tl.load(y_ptr + row * N + cols)
    w = tl.load(weight_ptr + cols)
    b = tl.load(bias_ptr + cols)

    # Fused add — promote to float32 for accuracy
    s = x.to(tl.float32) + y.to(tl.float32)

    # Store residual sum back to original dtype
    if IS_BF16:
        tl.store(sum_ptr + row * N + cols, s.to(tl.bfloat16))
    else:
        tl.store(sum_ptr + row * N + cols, s.to(tl.float16))

    # Layer-norm over the N=1024 dimension
    mean = tl.sum(s, axis=0) / BLOCK_N
    diff = s - mean
    var  = tl.sum(diff * diff, axis=0) / BLOCK_N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    norm = diff * rstd

    # Affine transform
    out = norm * w.to(tl.float32) + b.to(tl.float32)

    # Store LN output in original dtype
    if IS_BF16:
        tl.store(out_ptr + row * N + cols, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row * N + cols, out.to(tl.float16))


@torch.fx.wrap
def fused_add_layernorm_ln_first(bias, weight, x, y):
    """Fused add + layer_norm — returns (ln_out, sum)."""
    N       = 1024
    M       = x.numel() // N
    is_bf16 = (x.dtype == torch.bfloat16)

    sum_out = torch.empty_like(x)
    ln_out  = torch.empty_like(x)

    _fused_add_layernorm_lnf_kernel[(M,)](
        x, y, weight, bias,
        sum_out, ln_out,
        N,
        IS_BF16=is_bf16,
        BLOCK_N=1024,
        num_warps=8,
    )

    return (ln_out, sum_out)


# ---------------------------------------------------------------------------
# Pattern:  x + y  →  layer_norm  — returns (ln_out, sum)   [galsenai order]
# ---------------------------------------------------------------------------
def pattern(bias, weight, x, y):
    tmp_2 = x + y
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), weight, bias, 1e-05)
    return (tmp_4, tmp_2)


def replacement_args(bias, weight, x, y):
    return (bias, weight, x, y)


def replacement_func():
    return fused_add_layernorm_ln_first