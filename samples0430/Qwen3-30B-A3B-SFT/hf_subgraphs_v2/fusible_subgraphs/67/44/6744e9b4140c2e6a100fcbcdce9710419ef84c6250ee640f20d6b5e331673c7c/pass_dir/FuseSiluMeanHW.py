import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match mean((2,3)) over spatial dims of the silu output.
# The silu is left in the graph; we only replace the mean reduction.
# ---------------------------------------------------------------------------

def pattern(in_1):
    tmp_1 = in_1.mean((2, 3))
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: mean over H×W for each channel, no autotune overhead.
# Grid: (B*C,)  BLOCK_HW = 1024 covers all HW sizes in this problem.
# ---------------------------------------------------------------------------

@triton.jit
def _mean_hw_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    mean_val = total / HW
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean_hw(in_1):
    B, C, H, W = in_1.shape
    HW = H * W
    out = torch.empty((B, C), dtype=in_1.dtype, device=in_1.device)
    _mean_hw_kernel[(B * C,)](in_1, out, HW, BLOCK_HW=1024)
    return out


def replacement_func():
    return triton_mean_hw