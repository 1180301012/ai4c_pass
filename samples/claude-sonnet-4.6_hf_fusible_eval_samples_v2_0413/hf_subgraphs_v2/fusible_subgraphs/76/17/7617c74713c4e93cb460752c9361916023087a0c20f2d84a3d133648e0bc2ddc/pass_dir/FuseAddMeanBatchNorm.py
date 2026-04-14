import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helper: next power of 2 >= n  (pure Python, safe for concrete ints)
# ---------------------------------------------------------------------------
def _next_pow2(n) -> int:
    """Return the smallest power of 2 >= n.  Falls back to 256 for non-int."""
    if not isinstance(n, int):
        return 256
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Triton kernel: fuse (in_5 + in_4).mean(2,3) in one reduction pass.
#
# Grid: one program per (batch, channel) → B * C programs total.
# BLOCK_HW must be a power-of-2 >= HW.
# ---------------------------------------------------------------------------
@triton.jit
def add_mean_kernel(
    in4_ptr, in5_ptr,   # [B, C, H, W]
    out_ptr,            # [B, C]
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)   # in [0, B*C)
    base = pid * HW
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    x4 = tl.load(in4_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x5 = tl.load(in5_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_val = tl.sum(x4 + x5, axis=0) / HW   # fp32

    tl.store(out_ptr + pid, mean_val)           # Triton auto-casts to output dtype


# ---------------------------------------------------------------------------
# Host wrapper – @torch.fx.wrap makes FX treat this as an opaque leaf node
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_mean(in_5, in_4):
    """
    Replaces:
        tmp_4 = in_5 + in_4
        tmp_5 = tmp_4.mean((2, 3), keepdim=False)
        return tmp_5
    """
    B  = in_4.shape[0]
    C  = in_4.shape[1]
    HW = in_4.shape[2] * in_4.shape[3]

    BLOCK_HW = _next_pow2(HW)   # 49→64, 64→64, 144→256

    out = torch.empty((B, C), dtype=in_4.dtype, device=in_4.device)

    add_mean_kernel[(B * C,)](
        in_4, in_5,
        out,
        C, HW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_5, in_4):
    """
    Diagnostic minimal pattern: match only the add + spatial mean.
    Both in_4 and in_5 are [B, C, H, W]; output is [B, C].
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_5, in_4):
    return (in_5, in_4)


def replacement_func():
    # fused_add_mean is @torch.fx.wrap so the framework inserts it as a leaf.
    return fused_add_mean