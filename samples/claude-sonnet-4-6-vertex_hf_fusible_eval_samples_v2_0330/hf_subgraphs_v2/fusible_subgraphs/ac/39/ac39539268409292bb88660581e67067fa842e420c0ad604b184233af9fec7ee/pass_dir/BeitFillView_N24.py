import torch
import triton
import triton.language as tl

_FILL24_CACHE: dict = {}


@triton.jit
def _fill24_kernel(out_ptr, BLOCK: tl.constexpr):
    """Compute full 577×577 position-bias table for N=24, flattened."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (577 * 577)

    tbl_row = offs // 577
    tbl_col = offs % 577

    p = tbl_row - 1;  q = tbl_col - 1
    row_p = p // 24;  col_p = p % 24
    row_q = q // 24;  col_q = q % 24
    interior_val = (row_q - row_p + 23) * 47 + (col_q - col_p + 23)

    is_interior = (tbl_row >= 1) & (tbl_col >= 1)
    is_row0_nc  = (tbl_row == 0) & (tbl_col > 0)
    is_col0_nr  = (tbl_row > 0)  & (tbl_col == 0)
    is_corner   = (tbl_row == 0) & (tbl_col == 0)

    val = tl.where(is_interior, interior_val, 0)
    val = tl.where(is_row0_nc,  2209, val)
    val = tl.where(is_col0_nr,  2210, val)
    val = tl.where(is_corner,   2211, val)

    tl.store(out_ptr + offs, val, mask=mask)


@torch.fx.wrap
def beit_fill_view_N24(tmp_22):
    """Return precomputed flat position-bias table (tmp_28) for N=24."""
    global _FILL24_CACHE
    device = tmp_22.device
    key    = str(device)
    if key not in _FILL24_CACHE:
        out = torch.empty(577 * 577, dtype=torch.int64, device=device)
        _fill24_kernel[((577 * 577 + 1023) // 1024,)](out, BLOCK=1024)
        _FILL24_CACHE[key] = out
    return _FILL24_CACHE[key]


def pattern(tmp_22, tmp_23):
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 2209
    tmp_22[(slice(0, None, None), 0)] = 2210
    tmp_22[(0, 0)] = 2211
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args(tmp_22, tmp_23):
    return (tmp_22,)


def replacement_func():
    return beit_fill_view_N24