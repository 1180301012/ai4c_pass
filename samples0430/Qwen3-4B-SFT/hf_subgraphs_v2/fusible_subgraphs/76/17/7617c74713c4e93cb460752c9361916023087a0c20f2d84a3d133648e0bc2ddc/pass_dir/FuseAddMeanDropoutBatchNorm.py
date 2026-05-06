import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – single-output, single-arg BN inference.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, tmp_7):
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, tmp_7):
    return (in_0, in_1, in_2, in_3, tmp_7)


# ---------------------------------------------------------------------------
# Kernel: 2D grid (B, C), one CTA per (batch, channel) pair.
#
# Two specialisations are compiled implicitly by Triton on first call:
#   BLOCK_HW=64  – for HW ≤ 64  (HW=49 = 7×7)
#   BLOCK_HW=256 – for HW ≤ 144 (HW=64 = 8×8, HW=144 = 12×12)
#
# num_warps=4 ⟹ 128 threads per CTA. For BLOCK_HW=64: each thread handles
# 64/128 = 0.5 elements → 2 threads per 4-byte load → float2 vectorization.
# For BLOCK_HW=256: 2 elements per thread → float4 vectorization.
# ---------------------------------------------------------------------------
@triton.jit
def _bn_infer_kernel(
    x_ptr,    # [B, C]  flat input mean
    rMu_ptr,  # [C]
    rVa_ptr,  # [C]
    wt_ptr,   # [C]
    bs_ptr,   # [C]
    out_ptr,  # [B, C]  flat output
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    b = tl.program_id(0)   # batch index
    c = tl.program_id(1)   # channel index

    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    # Load & accumulate in fp32
    base = (b * C + c) * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x)

    # Load BN params (broadcast over batch)
    rm = tl.load(rMu_ptr + c).to(tl.float32)
    rv = tl.load(rVa_ptr + c).to(tl.float32)
    w  = tl.load(wt_ptr  + c).to(tl.float32)
    b_ = tl.load(bs_ptr  + c).to(tl.float32)

    # BN: fp32 → auto-cast to output dtype via pointer
    out = (s - rm) * tl.rsqrt(rv + 1e-5) * w + b_
    tl.store(out_ptr + base + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper – @torch.fx.wrap, single output.
# BLOCK_HW chosen to match HW exactly:
#   49 (7×7) → 64  (128 threads, each handles ≥1 element → float2/vectorised)
#   64 (8×8) → 256 (128 threads → 2 elems/thread → float4/vectorised)
#   144(12×12)→ 256
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _triton_bn_infer(in_0, in_1, in_2, in_3, tmp_7):
    """
    in_0=running_mean, in_1=running_var, in_2=bias, in_3=weight
    tmp_7: [B, C] (mean before BN)  returns [B, C] normalised.
    """
    B  = tmp_7.shape[0]
    C  = tmp_7.shape[1]
    HW = tmp_7.shape[2] * tmp_7.shape[3] if tmp_7.dim() == 4 else 1

    out_bn = torch.empty_like(tmp_7)

    # Smallest power-of-2 BLOCK_HW that covers HW – ensures no wasted lanes.
    # num_warps=4 → 128 threads; paired with 64/128 each handle ≤1 → both
    # give good ILP via float2/fp32 vector loads.
    if HW <= 64:
        BLOCK_HW = 64
    else:
        BLOCK_HW = 256

    _bn_infer_kernel[(B, C)](
        tmp_7, in_0, in_1, in_3, in_2,
        out_bn,
        C, HW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out_bn


def replacement_func():
    return _triton_bn_infer