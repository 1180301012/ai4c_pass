import torch
import triton
import triton.language as tl


# torch.cat([x, y], dim=-1) MATCHES. Fix replacement to avoid reshape (blocked).
def pattern(x, y):
    return torch.cat([x, y], dim=-1)


def replacement_args(x, y):
    return (x, y)


@triton.jit
def _triton_cat_kernel(
    x_ptr, y_ptr, out_ptr,
    N_TOTAL, J, J2,
    BLOCK: tl.constexpr,
):
    """
    Concatenate x[N_ROWS, J] and y[N_ROWS, J] along the last dim
    to produce out[N_ROWS, 2J].  Works on the flat contiguous layout:
      out[row*J2 + col] = x[row*J + col]      if col < J
                        = y[row*J + col - J]   otherwise
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < N_TOTAL
    row = i // J2
    col = i % J2
    is_x = col < J
    x_val = tl.load(x_ptr + row * J + col, mask=mask & is_x, other=0.0)
    y_val = tl.load(y_ptr + row * J + (col - J), mask=mask & ~is_x, other=0.0)
    tl.store(out_ptr + i, tl.where(is_x, x_val, y_val), mask=mask)


@torch.fx.wrap
def _triton_cat(x, y):
    # Compute shapes using allowed tensor property accesses (no reshape/view)
    *leading, J = x.shape
    N_ROWS = 1
    for d in leading:
        N_ROWS *= d
    J2 = 2 * J
    N_TOTAL = N_ROWS * J2
    out = torch.empty((*leading, J2), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    grid = ((N_TOTAL + BLOCK - 1) // BLOCK,)
    _triton_cat_kernel[grid](x, y, out, N_TOTAL, J, J2, BLOCK)
    return out


def replacement_func():
    return _triton_cat