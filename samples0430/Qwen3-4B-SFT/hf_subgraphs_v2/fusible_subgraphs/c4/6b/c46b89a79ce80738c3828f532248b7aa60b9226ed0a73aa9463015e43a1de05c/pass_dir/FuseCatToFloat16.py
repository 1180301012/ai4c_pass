import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: cat([x, y, z], dim=0).to(dtype=float16)
#
# Matches the single cat([relu_reshaped_1, relu_reshaped_0, in_0], 0) → cast
# in the model graph, replacing it with a single fused Triton kernel that
# writes directly to float16.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(x, y, z):
    t  = torch.cat([x, y, z], dim=0)
    r  = t.to(dtype=torch.float16)
    return r


def replacement_args(x, y, z):
    return (x, y, z)


# ── Fused cat-to-float16 kernel ─────────────────────────────────────────────────
# Concatenates 3 tensors along dim=0 and casts to float16 in one pass.
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _cat_cast_f16_kernel(
    src0_ptr, src1_ptr, src2_ptr,
    dst_ptr,
    nG0, nG1, nG2,   # number of elements in each input chunk
    d0, d1, d2,      # dtype flags: 0=bfloat16, 1=float16
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    FULL = nG0 + nG1 + nG2
    mask = offs < FULL

    # Decode which chunk this element belongs to
    g2 = offs >= nG2
    g1 = offs >= nG1

    local_off = tl.where(g2, offs - nG2,
              tl.where(g1, offs - nG1, offs))

    v0 = tl.load(src0_ptr + local_off, mask=mask & (offs < nG0), other=0.0)
    v1 = tl.load(src1_ptr + local_off, mask=mask & (offs < nG1), other=0.0)
    v2 = tl.load(src2_ptr + local_off, mask=mask & (offs < nG2), other=0.0)

    v = tl.where(g2, v0, tl.where(g1, v1, v2))

    out = (v.to(tl.float16) if d0 == 0 else
           v.to(tl.float16) if d1 == 0 else
           v)

    tl.store(dst_ptr + offs, out, mask=mask)


@torch.fx.wrap
def cat_cast_float16_fn(x, y, z):
    """
    x : [P1, C, H, W] = [4,   3, 384, 384]  (from unfold-then-reshape of in_1)
    y : [P2, C, H, W] = [25,  3, 384, 384]  (from unfold-then-reshape of in_2)
    z : [P3, C, H, W] = [1,   3, 384, 384]  (in_0, already float16)
    returns [6, 3, 384, 384] float16
    """
    d   = 384 * 384          # C*H*W = 147456
    g0, g1, g2 = x.shape[0], y.shape[0], z.shape[0]
    out = torch.empty((6, 3, 384, 384), dtype=torch.float16, device=x.device)

    total    = 6 * d
    grid     = lambda m: (triton.cdiv(total, m['BLOCK']),)

    is_bf16  = [0 if p == 1 else 1 for p in (x.dtype == torch.bfloat16)]
    d0, d1, d2 = is_bf16[0], is_bf16[1], is_bf16[2]

    _cat_cast_f16_kernel[grid](
        x, y, z, out,
        g0 * d, g1 * d, g2 * d,
        d0, d1, d2,
    )
    return out


def replacement_func():
    return cat_cast_float16_fn