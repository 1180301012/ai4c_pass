import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused (in2 + in3) / 2  →  LayerNorm  (single pass)
#
#  • BLOCK_SIZE=1024  (next power-of-2 ≥ N=768)
#  • Single-sweep: load x + y once, compute z, two reductions, affine store.
#  • Masked elements (768..1023) become 0.0 via "other=0.0" so the sums
#    are exact: mean = sum(x[0..767]) / 768,  var = E[z²] - mean².
#  • No loop, no autotune — minimal dispatch overhead.
# ---------------------------------------------------------------------------
@triton.jit
def fused_add_half_layernorm_kernel(
    in0_ptr,    # bias   [768]
    in1_ptr,    # weight [768]
    in2_ptr,    # [rows, 768]
    in3_ptr,    # [rows, 768]
    out_ptr,    # [rows, 768]
    BLOCK_SIZE: tl.constexpr,   # 1024
    eps:        tl.constexpr,   # 1e-12
):
    row     = tl.program_id(0)
    row_off = row * 768
    offs    = tl.arange(0, BLOCK_SIZE)
    mask    = offs < 768

    x = tl.load(in2_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(in3_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
    z = (x + y) * 0.5                    # masked → 0.0

    # Two reductions over the 1024-element vector
    # (masked elements carry 0.0 → no contribution to sums)
    sum_z  = tl.sum(z,        axis=0)
    sum_z2 = tl.sum(z * z,    axis=0)
    mean   = sum_z  / 768
    var    = sum_z2 / 768 - mean * mean
    rstd   = tl.rsqrt(var + eps)

    w  = tl.load(in1_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bv = tl.load(in0_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + row_off + offs, (z - mean) * rstd * w + bv, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper  (@torch.fx.wrap required)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_half_layernorm(in_0, in_1, in_2, in_3):
    rows = in_2.shape[0]
    out  = torch.empty_like(in_2)
    fused_add_half_layernorm_kernel[(rows,)](
        in_0, in_1, in_2, in_3, out,
        BLOCK_SIZE=1024,
        eps=1e-12,
        num_warps=4,
        num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_half_layernorm