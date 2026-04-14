import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass: replace  x.transpose(-2, -1)  with a Triton tiled-transpose kernel.
# Routing tag "transpose" lets us share replacement_func with
# FuseScaleTranspose_0_1767766952966369.py (same _shared_dispatch object).
# ---------------------------------------------------------------------------
def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x, "transpose")


# ---------------------------------------------------------------------------
# Scale kernel – needed so the dispatcher below can be fully defined here
# (the elif branch for "scale" is never executed in this pass's context)
# ---------------------------------------------------------------------------
@triton.jit
def _scale_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) * 0.1767766952966369,
             mask=mask)


# ---------------------------------------------------------------------------
# Transpose kernel
# ---------------------------------------------------------------------------
@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    M, N,
    TILE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_start = pid_m * TILE
    col_start = pid_n * TILE

    rows = row_start + tl.arange(0, TILE)
    cols = col_start + tl.arange(0, TILE)

    mask = (rows[:, None] < M) & (cols[None, :] < N)

    in_off  = pid_b * M * N + rows[:, None] * N + cols[None, :]
    tile    = tl.load(x_ptr + in_off, mask=mask, other=0.0)

    out_off = pid_b * N * M + cols[None, :] * M + rows[:, None]
    tl.store(out_ptr + out_off, tile, mask=mask)


# ---------------------------------------------------------------------------
# SHARED dispatcher – identical to the one in FuseScaleTranspose_0_1767....py
# so the framework sees ONE unique replacement function across both passes.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _shared_dispatch(x, route):
    if route == "scale":
        n          = x.numel()
        BLOCK_SIZE = 512
        nblocks    = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        out        = torch.empty_like(x)
        _scale_kernel[(nblocks,)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
    elif route == "transpose":
        s    = x.shape
        B    = x.numel() // (s[-2] * s[-1])
        M    = s[-2]
        N    = s[-1]
        TILE = 32
        gm   = (M + TILE - 1) // TILE
        gn   = (N + TILE - 1) // TILE
        out  = torch.empty(s[0], s[1], N, M, dtype=x.dtype, device=x.device)
        _transpose_kernel[(B, gm, gn)](x, out, M, N, TILE=TILE)
        return out
    # unreachable fallback
    return x


def replacement_func():
    return _shared_dispatch