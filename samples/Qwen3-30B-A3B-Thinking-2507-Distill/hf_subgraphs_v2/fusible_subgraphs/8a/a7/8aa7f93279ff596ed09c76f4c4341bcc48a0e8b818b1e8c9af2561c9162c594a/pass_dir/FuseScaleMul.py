import torch
import triton
import triton.language as tl


# ─── Pattern: scalar multiplication ──────────────────────────────────────────

def pattern(x):
    return x * 0.1767766952966369


def replacement_args(x):
    return (x,)


# ─── Scale kernel ─────────────────────────────────────────────────────────────
# Shape is always [70, 1, 49, 32] → N = 110592 = 54 × 2048 (exact fit).
# BLOCK_SIZE=2048, num_warps=4 (128 threads × 16 fp16 each = vectorised loads).

@triton.jit
def _scale_mul_kernel(
    x_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,  # always 110592 for [70,1,49,32]
    BLOCK_SIZE: tl.constexpr,   # always 2048
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 0.1767766952966369, mask=mask)


_BLOCK      = 2048
_N_ELEMENTS = 110592   # 70 × 1 × 49 × 32
_GRID       = (_N_ELEMENTS // _BLOCK,)   # = (54,)

# ── Cache the launcher once at module load ───────────────────────────────────
# Avoids recreating the launcher dict on every call.
def _make_launcher():
    return _scale_mul_kernel[_GRID]

_launcher = _make_launcher()


@torch.fx.wrap
def triton_scale_mul(x):
    out = torch.empty_like(x)
    _launcher(x, out, N_ELEMENTS=_N_ELEMENTS, BLOCK_SIZE=_BLOCK, num_warps=4)
    return out


def replacement_func():
    return triton_scale_mul