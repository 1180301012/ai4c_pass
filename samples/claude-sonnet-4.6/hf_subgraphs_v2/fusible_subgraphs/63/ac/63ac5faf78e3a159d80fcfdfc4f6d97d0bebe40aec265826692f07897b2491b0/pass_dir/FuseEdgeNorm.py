import torch
import triton
import triton.language as tl


@triton.jit
def _gather_mul_kernel(
    norm_ptr,  # [N] masked-norm array  (deg^{-0.5} with inf→0)
    row_ptr,   # [E] row indices, int64
    col_ptr,   # [E] col indices, int64
    ew_ptr,    # [E] edge weights, float16 / bfloat16
    out_ptr,   # [E] output, same dtype as ew
    E,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: out[e] = norm[row[e]] * ew[e] * norm[col[e]]
    Replaces  getitem(tmp_2, row) * edge_weight * getitem(tmp_2, col)
    in one kernel launch, operating natively in float16/bfloat16.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < E

    ri = tl.load(row_ptr + offs, mask=mask, other=0)
    ci = tl.load(col_ptr + offs, mask=mask, other=0)

    ew = tl.load(ew_ptr   + offs, mask=mask, other=0.0)
    nr = tl.load(norm_ptr + ri,   mask=mask, other=0.0)
    nc = tl.load(norm_ptr + ci,   mask=mask, other=0.0)

    tl.store(out_ptr + offs, nr * ew * nc, mask=mask)


@torch.fx.wrap
def _gather_mul_fn(tmp_2, in_4, in_5, in_2):
    """
    Fused replacement for gather×2 + mul×2:
      tmp_5 = tmp_2[in_5]
      tmp_6 = tmp_5 * in_4
      tmp_7 = tmp_2[in_2]
      tmp_8 = tmp_6 * tmp_7

    tmp_2 is the masked degree-normalisation array [N] already produced by the
    upstream pow_ + __eq__ + masked_fill_ chain.
    """
    E = in_4.numel()
    out = torch.empty_like(in_4)
    BLOCK_SIZE = 256
    grid = ((E + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _gather_mul_kernel[grid](
        tmp_2,  # norm_ptr
        in_5,   # row_ptr
        in_2,   # col_ptr
        in_4,   # ew_ptr
        out,    # out_ptr
        E,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(tmp_2, in_4, in_5, in_2):
    """
    Match the live data-flow chain: gather×2 + mul×2.

    tmp_2 is a PLACEHOLDER (exempt from containment checks) so the
    upstream pow_ / __eq__ / masked_fill_ nodes that also consume it
    in the model graph cause no containment violation.
    """
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_4, in_5, in_2):
    return (tmp_2, in_4, in_5, in_2)


def replacement_func():
    return _gather_mul_fn