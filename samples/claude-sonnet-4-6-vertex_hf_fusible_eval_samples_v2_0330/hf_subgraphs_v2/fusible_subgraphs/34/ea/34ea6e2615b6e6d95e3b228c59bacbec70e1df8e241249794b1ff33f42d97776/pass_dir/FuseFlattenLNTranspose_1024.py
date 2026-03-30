import torch
import triton
import triton.language as tl


# ── Pattern: exactly 1 returning node ────────────────────────────────────────
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# C=1024=BLOCK_C → no masking needed. float32 throughout.
# Single-pass variance (E[x²]−E[x]²).

@triton.jit
def _ln_f32_1024(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_n,
    stride_c,
    N,
    C,
    eps,
    BLOCK_C: tl.constexpr,
):
    n    = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)   # no mask: BLOCK_C == C == 1024

    x = tl.load(x_ptr + n * stride_n + cols * stride_c)

    mean = tl.sum(x, axis=0)      / C
    var  = tl.sum(x * x, axis=0)  / C - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w    = tl.load(w_ptr + cols)
    b_   = tl.load(b_ptr + cols)
    out  = (x - mean) * rstd * w + b_
    tl.store(y_ptr + n * C + cols, out)


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def opt_ln_1024(x, weight, bias):
    B, N, C = x.shape
    sn = x.stride(1)
    sc = x.stride(2)
    y  = torch.empty(B, N, C, dtype=x.dtype, device=x.device)
    _ln_f32_1024[(B * N,)](x, weight, bias, y, sn, sc, N, C, 1e-5,
                           BLOCK_C=1024, num_warps=4)
    return y


def replacement_func():
    return opt_ln_1024