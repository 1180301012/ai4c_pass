import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: mean over spatial (H,W) dimension.
# No autotune — each HW value gets one pre-determined BLOCK_SIZE chosen to
# minimise wasted loads and maximise GPU occupancy.
# ---------------------------------------------------------------------------

@triton.jit
def _mean_kernel(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    vals = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals)
    mean_f32 = total / HW

    tl.store(out_ptr + pid, mean_f32)


# ---------------------------------------------------------------------------
# Wrapper — dispatches to the right compiled variant by HW
# ---------------------------------------------------------------------------

_BLOCK_SIZE_MAP = {
    # HW → (BLOCK_SIZE, num_warps)
    # Sorted keys so lookup picks the smallest BLOCK_SIZE that covers HW.
    64:   (64,   2),   # exact fit,  240 CTAs → full occupancy
    196:  (256,  4),   # 1 load, default warp count
    256:  (256,  4),   # exact fit
    576:  (1024, 4),   # 1 load
    784:  (1024, 4),   # 1 load
    1024: (1024, 4),   # exact fit
    2304: (4096, 4),   # 1 load
}


def _pick_block_size(HW: int):
    """Return (BLOCK_SIZE, num_warps) for a given HW."""
    for k in sorted(_BLOCK_SIZE_MAP.keys()):
        if HW <= k:
            return _BLOCK_SIZE_MAP[k]
    return (4096, 8)


@torch.fx.wrap
def fused_slice_mean(tmp_1):
    """
    Replaces: tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    tmp_1: [B, C, H, W]  →  out: [B, C, 1, 1]
    """
    B, C, H, W = tmp_1.shape
    HW = H * W
    BC = B * C

    out_mean = torch.empty((B, C, 1, 1), dtype=tmp_1.dtype, device=tmp_1.device)

    BLOCK_SIZE, num_warps = _pick_block_size(HW)
    _mean_kernel[(BC,)](
        tmp_1,
        out_mean,
        HW,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out_mean


# ---------------------------------------------------------------------------
# Pattern / replacement wiring
# ---------------------------------------------------------------------------

def pattern(tmp_1):
    """Match: tmp_2 = tmp_1.mean((2, 3), keepdim=True)"""
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_2


def replacement_args(tmp_1):
    return (tmp_1,)


def replacement_func():
    return fused_slice_mean