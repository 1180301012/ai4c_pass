import torch
import triton
import triton.language as tl

_CACHE = {}


@triton.jit
def _relpos_kernel(out_ptr, n_side, mul, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    grid_n = n_side * n_side
    base = offs - 1

    is_zero = offs == 0
    in_body = offs > 0

    row = base // (n_side * n_side)
    col = base % (n_side * n_side)
    r1 = row // n_side
    c1 = row % n_side
    r2 = col // n_side
    c2 = col % n_side

    val = ((r1 - r2 + (n_side - 1)) * mul + (c1 - c2 + (n_side - 1))).to(tl.int64)
    val = tl.where(in_body, val, 0)
    val = tl.where(offs < (grid_n + 1), 3970, val)
    val = tl.where((offs % (grid_n + 1)) == 0, 3969, val)
    val = tl.where(is_zero, 3971, val)

    tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def _relpos_kernel_generic(out_ptr, n_side, mul, row_fill, col_fill, corner_fill, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    grid_n = n_side * n_side
    base = offs - 1

    is_zero = offs == 0
    in_body = offs > 0

    row = base // (n_side * n_side)
    col = base % (n_side * n_side)
    r1 = row // n_side
    c1 = row % n_side
    r2 = col // n_side
    c2 = col % n_side

    val = ((r1 - r2 + (n_side - 1)) * mul + (c1 - c2 + (n_side - 1))).to(tl.int64)
    val = tl.where(in_body, val, 0)
    val = tl.where(offs < (grid_n + 1), col_fill, val)
    val = tl.where((offs % (grid_n + 1)) == 0, row_fill, val)
    val = tl.where(is_zero, corner_fill, val)

    tl.store(out_ptr + offs, val, mask=mask)


def _make_relpos_tensor(n_side: int, device: torch.device):
    total = (n_side * n_side + 1) * (n_side * n_side + 1)
    out = torch.empty((total,), dtype=torch.int64, device=device)
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
    mul = 2 * n_side - 1
    if n_side == 32:
        _relpos_kernel[grid](out, n_side, mul, total, BLOCK=1024)
    else:
        row_fill = (2 * n_side - 1) ** 2
        col_fill = row_fill + 1
        corner_fill = row_fill + 2
        _relpos_kernel_generic[grid](out, n_side, mul, row_fill, col_fill, corner_fill, total, BLOCK=1024)
    return out


def _get_cached_relpos(n_side: int, device: torch.device):
    key = (n_side, device.type, device.index)
    out = _CACHE.get(key)
    if out is None or out.device != device:
        out = _make_relpos_tensor(n_side, device)
        _CACHE[key] = out
    return out


@torch.fx.wrap
def beit_relpos_dispatch(in_0, in_1, route: str):
    out0 = torch.cat([in_1, in_0])
    if route == 'n32':
        out1 = _get_cached_relpos(32, out0.device)
    elif route == 'n24':
        out1 = _get_cached_relpos(24, out0.device)
    elif route == 'n14':
        out1 = _get_cached_relpos(14, out0.device)
    else:
        raise RuntimeError(f'Unknown route: {route}')
    return out0, out1