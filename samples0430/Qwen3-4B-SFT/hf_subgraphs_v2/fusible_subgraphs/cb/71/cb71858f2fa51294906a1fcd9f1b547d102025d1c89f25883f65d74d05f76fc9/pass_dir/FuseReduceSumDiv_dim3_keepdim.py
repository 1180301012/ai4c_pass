import torch
import triton
import triton.language as tl


# ─── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_3):
    """
    Matches:
        tmp_5 = in_3.sum(dim=3, keepdim=True)        # reduce last dim
        tmp_6 = in_3 / tmp_5                         # normalize
    """
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    return (in_3,)


# ─── Triton kernel ─────────────────────────────────────────────────────────────
#
# 2-D tile per program: shape [ROWS_PER_PROG, RED_N].
#   tl.sum(x, axis=1)  reduces along the RED_N dim → [ROWS_PER_PROG].
#   x / x_sum[:,None]  is pure 2-D  [ROWS, RED_N] / [ROWS, 1]  — no 3-D shape.

@triton.jit
def sum_div_kernel(
    in_ptr,
    out_ptr,
    RED_N:     tl.constexpr,   # power-of-2 >= N
    NUM_ROWS:  tl.constexpr,   # B*C*H
):
    pid  = tl.program_id(0)
    base = pid * RED_N
    cols = tl.arange(0, RED_N)

    x = tl.load(in_ptr + base + cols).to(tl.float32)

    x_sum = tl.sum(x, axis=0)           # scalar (no guard needed: grid exact)

    result = (x / x_sum).to(in_ptr.dtype.element_ty)
    tl.store(out_ptr + base + cols, result)


@torch.fx.wrap
def triton_sum_div(in_3):
    B, C, H, N = in_3.shape
    RED_N   = 8    # N=8
    NUM_ROWS = B * C * H
    out     = torch.empty_like(in_3)

    sum_div_kernel[(NUM_ROWS,)](
        in_3, out,
        RED_N=RED_N,
        NUM_ROWS=NUM_ROWS,
    )
    return out


# ─── Replacement ───────────────────────────────────────────────────────────────

def replacement_func():
    return triton_sum_div