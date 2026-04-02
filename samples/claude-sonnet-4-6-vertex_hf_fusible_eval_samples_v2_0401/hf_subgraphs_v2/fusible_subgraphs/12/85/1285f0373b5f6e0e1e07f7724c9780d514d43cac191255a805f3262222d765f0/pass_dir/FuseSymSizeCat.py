"""
Pass: FuseSymSizeCat

Matches:  sym_size(a, 1)  +  cat([a, b], dim=1)
          (two separate nodes that both consume the same tensor 'a')

Returns:  (M, cat_out)  where M = a.shape[1] plays the role of sym_size's output.

By matching sym_size together with cat we eliminate the sym_size FX-node
dispatch and merge it into a single replacement call, slightly reducing the
number of Python operations that stall the GPU stream between kernel launches.
"""
import torch
import triton
import triton.language as tl

if not hasattr(torch, 'sym_sum'):
    torch.sym_sum = sum


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: sym_size(a, 1)  +  cat([a, b], dim=1)
# Both outputs are observable outside the matched subgraph:
#   - sym_size result (used by assertion / sym_sum nodes)
#   - cat result (returned by the model)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(a, b):
    size_a = torch.ops.aten.sym_size.int(a, 1)
    result = torch.cat([a, b], dim=1)
    return size_a, result


def replacement_args(a, b):
    return (a, b)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel (2-D grid, one warp per block)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def sym_cat_kernel(
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
# Replacement wrapper – returns (M, cat_out) matching (size_a, result) above
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def triton_sym_size_cat(a, b):
    if not a.is_cuda:
        a = a.to(b.device)

    rows     = a.shape[0]
    cols_a   = a.shape[1]          # this IS sym_size(a, 1)
    cols_b   = b.shape[1]
    cols_out = cols_a + cols_b

    out = torch.empty((rows, cols_out), dtype=a.dtype, device=b.device)

    sym_cat_kernel[
        (rows, triton.cdiv(cols_out, 32))
    ](
        a, b, out,
        cols_a, cols_b,
        BLOCK=32,
        num_warps=1,
    )

    return cols_a, out   # (M = sym_size result,  cat output)


def replacement_func():
    return triton_sym_size_cat