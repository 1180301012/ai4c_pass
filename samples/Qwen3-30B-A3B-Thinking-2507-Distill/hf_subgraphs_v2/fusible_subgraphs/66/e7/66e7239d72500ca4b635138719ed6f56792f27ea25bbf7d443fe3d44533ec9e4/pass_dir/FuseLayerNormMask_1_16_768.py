import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
# ---------------------------------------------------------------------------

def pattern(in_3, in_2, in_1):
    return torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)


def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)


# ---------------------------------------------------------------------------
# Triton layer-norm kernel
#   One program per row (N=16 programs for [1,16,768]).
#   BLOCK_H=768 exactly (non-power-of-2; no masking waste).
#   All reductions in fp32 for numerical stability.
# ---------------------------------------------------------------------------

@triton.jit
def _ln_kernel(
    x_ptr,   # [N, H]  fp16/bf16
    w_ptr,   # [H]     fp16/bf16 – weight
    b_ptr,   # [H]     fp16/bf16 – bias
    out_ptr, # [N, H]  fp16/bf16 – output
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Load row with mask (256 padding elements are 0.0)
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean (masked zeros don't contribute)
    mean = tl.sum(x, axis=0) / H

    # Variance
    xc  = x - mean
    var = tl.sum(xc * xc, axis=0) / H
    rstd = tl.rsqrt(var + eps)

    # Scale and shift
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = xc * rstd * w + b

    tl.store(out_ptr + row * H + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def triton_layer_norm(x, weight, bias):
    B, S, H = x.shape
    N = B * S
    out = torch.empty_like(x)
    _ln_kernel[(N,)](
        x, weight, bias, out,
        H, 1e-12,
        BLOCK_H=1024,
        num_warps=8,
    )
    return out


def replacement_func():
    return triton_layer_norm