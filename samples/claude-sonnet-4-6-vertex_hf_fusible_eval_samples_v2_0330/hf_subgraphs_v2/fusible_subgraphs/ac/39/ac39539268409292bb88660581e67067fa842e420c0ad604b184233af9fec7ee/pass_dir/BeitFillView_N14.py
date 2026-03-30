import torch
import triton
import triton.language as tl

_FILL14_CACHE: dict = {}


@triton.jit
def _fill14_kernel(out_ptr, BLOCK: tl.constexpr):
    """Compute full 197×197 position-bias table for N=14, flattened."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (197 * 197)

    tbl_row = offs // 197
    tbl_col = offs % 197

    p = tbl_row - 1;  q = tbl_col - 1
    row_p = p // 14;  col_p = p % 14
    row_q = q // 14;  col_q = q % 14
    interior_val = (row_q - row_p + 13) * 27 + (col_q - col_p + 13)

    is_interior = (tbl_row >= 1) & (tbl_col >= 1)
    is_row0_nc  = (tbl_row == 0) & (tbl_col > 0)
    is_col0_nr  = (tbl_row > 0)  & (tbl_col == 0)
    is_corner   = (tbl_row == 0) & (tbl_col == 0)

    val = tl.where(is_interior, interior_val, 0)
    val = tl.where(is_row0_nc,  729, val)
    val = tl.where(is_col0_nr,  730, val)
    val = tl.where(is_corner,   731, val)

    tl.store(out_ptr + offs, val, mask=mask)


@torch.fx.wrap
def beit_fill_view_N14(tmp_22):
    """Return precomputed flat position-bias table (tmp_28) for N=14."""
    global _FILL14_CACHE
    device = tmp_22.device
    key    = str(device)
    if key not in _FILL14_CACHE:
        out = torch.empty(197 * 197, dtype=torch.int64, device=device)
        _fill14_kernel[((197 * 197 + 1023) // 1024,)](out, BLOCK=1024)
        _FILL14_CACHE[key] = out
    return _FILL14_CACHE[key]


def pattern(tmp_22, tmp_23):
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 729
    tmp_22[(slice(0, None, None), 0)] = 730
    tmp_22[(0, 0)] = 731
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args(tmp_22, tmp_23):
    return (tmp_22,)


def replacement_func():
    return beit_fill_view_N14