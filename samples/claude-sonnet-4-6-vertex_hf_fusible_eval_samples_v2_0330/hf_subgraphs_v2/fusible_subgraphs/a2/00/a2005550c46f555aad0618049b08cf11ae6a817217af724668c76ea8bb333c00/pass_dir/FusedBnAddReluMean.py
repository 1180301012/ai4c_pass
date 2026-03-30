import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: match ONLY the mean (call_method, confirmed to work as anchor).
# tmp_6 is the relu output passed in as placeholder.
# Returns (tmp_6, mean_out) – mirrors model's return exactly.
# ─────────────────────────────────────────────────────────────────────────────

def pattern(tmp_6):
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return tmp_7


def replacement_args(tmp_6):
    return (tmp_6,)


# ─────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel: spatial mean over (H, W) dims with keepdim=True
# Grid: (N*C,)  – one program per (batch, channel) slice
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _spatial_mean_kernel(
    x_ptr,          # [N, C, H, W]  – input
    mean_out_ptr,   # [N*C]         – flat mean output
    HW,             # H * W
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW
    acc  = 0.0

    for block_start in range(0, HW, BLOCK_HW):
        idx  = block_start + tl.arange(0, BLOCK_HW)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        x_valid = tl.where(mask, x, tl.zeros_like(x))
        acc = acc + tl.sum(x_valid, axis=0)

    tl.store(mean_out_ptr + pid, (acc / HW).to(DTYPE))


_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@torch.fx.wrap
def triton_spatial_mean(tmp_6):
    """
    tmp_6: relu output  [N, C, H, W]
    Returns: spatial_mean [N, C, 1, 1]  (tmp_6 stays in graph via model's output)
    """
    N, C, H, W = tmp_6.shape
    HW = H * W
    mean_out = torch.empty((N, C, 1, 1), dtype=tmp_6.dtype, device=tmp_6.device)
    DTYPE = _DTYPE_MAP[tmp_6.dtype]
    _spatial_mean_kernel[(N * C,)](
        tmp_6, mean_out.view(N * C), HW, DTYPE,
        BLOCK_HW=1024, num_warps=4,
    )
    return mean_out


def replacement_func():
    return triton_spatial_mean