import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the L2 normalisation + division
#   tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
#   tmp_1 = in_1 / tmp_0
#   return tmp_1
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: one program per row.
#   Grid : (M,)   — one Triton program per row of the [M, N] input
#   BLOCK_N : power-of-2 >= N, minimum 128
#
#   For N=1024  (exact power-of-2): no masking needed, fully coalesced.
#   For N=768 / N=1152 : BLOCK_N=2048, masking on last 1280/896 elements.
# ---------------------------------------------------------------------------
@triton.jit
def _l2_norm_div_kernel(
    in_ptr,
    out_ptr,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(in_ptr + row * stride_m + offs,
                 mask=mask, other=0.0).to(tl.float32)

    # L2 norm (reduction over the N-element row)
    norm_sq = tl.sum(x * x, axis=0)
    norm    = tl.sqrt(norm_sq)
    out     = (x / norm).to(tl.bfloat16)

    tl.store(out_ptr + row * N + offs, out, mask=mask)


@torch.fx.wrap
def l2_norm_div(in_1):
    M, N = in_1.shape          # e.g. [2, 768], [2, 1024], [2, 1152]
    out  = torch.empty_like(in_1)

    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N < 128:
        BLOCK_N = 128

    _l2_norm_div_kernel[(M,)](
        in_1, out,
        N, in_1.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def replacement_func():
    return l2_norm_div