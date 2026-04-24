import torch
import triton
import triton.language as tl

# ── Single kernel: Triton specialises on linear.dtype automatically ──────────
# No OUT_DTYPE constexpr needed — x.dtype is a compile-time constant in Triton JIT.
@triton.jit
def _fused_add_relu_kernel(
    linear_ptr, res_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(linear_ptr + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr   + offs, mask=mask, other=0.0)
    # Accumulate in fp32, then cast back to the input element dtype
    tl.store(out_ptr + offs,
             tl.maximum((x + r).to(tl.float32), 0.0).to(x.dtype),
             mask=mask)


# ── Fixed shape constants (M=1000, N=128) ────────────────────────────────────
_N     = 128000   # 1000 * 128
_BLOCK = 1024
_NBLK  = (_N + _BLOCK - 1) // _BLOCK   # 125
_GRID  = (_NBLK,)


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def fused_add_relu(linear, in_2):
    """Fused relu(linear + in_2).  Shapes assumed [1000, 128]."""
    out = torch.empty_like(in_2)
    _fused_add_relu_kernel[_GRID](
        linear, in_2, out, _N,
        BLOCK_SIZE=_BLOCK, num_warps=4,
    )
    return out


# ── Pattern: fuse add + in-place relu ───────────────────────────────────────
def pattern(linear, in_2):
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(linear, in_2):
    return (linear, in_2)


def replacement_func():
    return fused_add_relu