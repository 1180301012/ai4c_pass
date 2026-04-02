import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────────────────────
# Compatibility shim: torch.sym_sum → Python built-in sum
# ──────────────────────────────────────────────────────────────────────────────
if not hasattr(torch, 'sym_sum'):
    torch.sym_sum = sum


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: torch.cat([a, b], dim=1)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(a, b):
    return torch.cat([a, b], dim=1)


def replacement_args(a, b):
    return (a, b)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: 2-D grid cat along dim=1 for 2-D int64 tensors.
#   pid(0) = row  (no integer division needed)
#   pid(1) = column-block
#   output[row, col] = a[row, col]           if col < cols_a
#                    = b[row, col-cols_a]    otherwise
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def cat_dim1_kernel(
    a_ptr, b_ptr, out_ptr,
    cols_a, cols_b,
    BLOCK: tl.constexpr,
):
    row     = tl.program_id(0)
    col_pid = tl.program_id(1)
    cols_out = cols_a + cols_b

    cols = col_pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < cols_out
    in_a = cols < cols_a

    a_vals = tl.load(a_ptr + row * cols_a + cols,            mask=(mask & in_a),  other=0)
    b_vals = tl.load(b_ptr + row * cols_b + (cols - cols_a), mask=(mask & ~in_a), other=0)
    tl.store(out_ptr + row * cols_out + cols, tl.where(in_a, a_vals, b_vals), mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Replacement wrapper
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_cat_dim1(a, b):
    if not a.is_cuda:
        a = a.to(b.device)

    rows     = a.shape[0]
    cols_a   = a.shape[1]
    cols_b   = b.shape[1]
    cols_out = cols_a + cols_b

    out = torch.empty((rows, cols_out), dtype=a.dtype, device=b.device)

    # BLOCK=32, num_warps=1 → one warp per block, maximises block count for
    # tiny tensors and keeps per-launch Python overhead minimal.
    cat_dim1_kernel[
        (rows, triton.cdiv(cols_out, 32))
    ](a, b, out, cols_a, cols_b, BLOCK=32, num_warps=1)

    return out


def replacement_func():
    return triton_cat_dim1