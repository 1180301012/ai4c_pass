import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: divide by 2.8284271247461903, then transpose last two dims
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: fused scale + transpose (2-D tiling with tl.trans)
#   - Reads  [BLOCK_S, BLOCK_D] from input   (stride-1 in D → coalesced reads)
#   - Writes [BLOCK_D, BLOCK_S] to output    (stride-1 in S → coalesced writes)
#   Grid: (B*H, cdiv(S, BLOCK_S))
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 16, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 32, 'BLOCK_D': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 64, 'BLOCK_D': 64}, num_warps=8, num_stages=3),
    ],
    key=['S', 'D'],
    warmup=5,
    rep=20,
)
@triton.jit
def _div_transpose_kernel_28284(
    in_ptr, out_ptr,
    S, D,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    s_tile = tl.program_id(1)
    s_base = s_tile * BLOCK_S

    s_offs = s_base + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    # ---- coalesced load [BLOCK_S, BLOCK_D] from input ----
    in_offs = bh * S * D + s_offs[:, None] * D + d_offs[None, :]
    x = tl.load(in_ptr + in_offs, mask=s_mask[:, None], other=0.0)

    # ---- apply scale ----
    x = x / scale

    # ---- register-level transpose → [BLOCK_D, BLOCK_S] ----
    x_t = tl.trans(x)

    # ---- coalesced store [BLOCK_D, BLOCK_S] to output ----
    out_offs = bh * D * S + d_offs[:, None] * S + s_offs[None, :]
    tl.store(out_ptr + out_offs, x_t, mask=s_mask[None, :])


# ---------------------------------------------------------------------------
# Grid cache: avoids recreating the lambda on every call
# ---------------------------------------------------------------------------
_GRID_CACHE_28284: dict = {}


@torch.fx.wrap
def _div_transpose_wrapper_28284(in_0):
    B, H, S, D = in_0.shape
    out = torch.empty(B, H, D, S, dtype=in_0.dtype, device=in_0.device)

    grid_key = (B, H, S, D)
    if grid_key not in _GRID_CACHE_28284:
        _GRID_CACHE_28284[grid_key] = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_S']))
    grid = _GRID_CACHE_28284[grid_key]

    _div_transpose_kernel_28284[grid](
        in_0, out,
        S, D,
        2.8284271247461903,
    )
    return out


def replacement_func():
    return _div_transpose_wrapper_28284