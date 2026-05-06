import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches  tmp_0 * tmp_1 -> sum(dim=1) -> 5 - result
#   • in_0 : softmax output  [B, N_COLS]
#   • in_1 : linspace weight [N_COLS]   (= torch.linspace(0, 4, 5))
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_2 = in_0 * in_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: for x[B, 5] and w[5] → result[B]
#   result[b] = 5 - sum( x[b,c] * w[c]  for c in 0..4 )
#   N_REAL=5 padded to NNGTH=8 (next power of 2)
# ---------------------------------------------------------------------------
@triton.jit
def fused_weighted_sum_subtract_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N_REAL: tl.constexpr,   # 5 actual columns
    NNGTH: tl.constexpr,    # 8  (next power of 2 >= N_REAL)
):
    pid = tl.program_id(0)
    cols = tl.arange(0, NNGTH)    # [0, 1, 2, 3, 4, 5, 6, 7]
    mask = cols < N_REAL

    row_base = pid * N_REAL

    x   = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    w   = tl.load(w_ptr      + cols,       mask=mask, other=0.0).to(tl.float32)

    result = tl.sum(x * w, axis=0)
    out = (5.0 - result).to(x_ptr.dtype.element_ty)

    tl.store(out_ptr + pid, out)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_weighted_sum_subtract(x, w):
    B       = x.shape[0]
    N_REAL  = x.shape[1]   # 5
    NNGTH   = 8             # next power of 2 >= 5

    out = torch.empty(B, dtype=x.dtype, device=x.device)

    fused_weighted_sum_subtract_kernel[(B,)](
        x_ptr=x,
        w_ptr=w,
        out_ptr=out,
        N_REAL=N_REAL,
        NNGTH=NNGTH,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg, returns the callable (do NOT call it)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_weighted_sum_subtract