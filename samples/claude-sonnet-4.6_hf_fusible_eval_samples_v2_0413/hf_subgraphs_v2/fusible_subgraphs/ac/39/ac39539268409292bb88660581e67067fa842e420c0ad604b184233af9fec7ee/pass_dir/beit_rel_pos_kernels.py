"""
Shared Triton kernels for BEiT relative position bias index computation.

The original computation:
  1. Creates [2, N, N] grid coordinates via arange+meshgrid+stack+flatten
  2. Broadcasts subtraction to produce [2, N², N²] difference tensor  (EXPENSIVE!)
  3. Permutes [N², N², 2] + contiguous copy  (EXPENSIVE!)
  4. In-place adds/multiply, sum(-1) → [N², N²] index table
  5. Fills (N²+1)×(N²+1) zeros matrix, then view(-1)

Our replacement: a single Triton kernel that directly computes the final
flat int64 index table without any intermediate [2, N², N²] tensor.

For each (i,j) in [0,N²) × [0,N²):
  ri = i // N,  ci = i % N
  rj = j // N,  cj = j % N
  index = (ri - rj + N-1) * (2N-1) + (ci - cj + N-1)

The output (N²+1)×(N²+1) matrix layout (flattened):
  [0,0]     = V2 = (2N-1)² + 2
  [0, j>0]  = V0 = (2N-1)²
  [i>0, 0]  = V1 = (2N-1)² + 1
  [i>0,j>0] = computed index
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["N"],
)
@triton.jit
def rel_pos_idx_kernel(
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # All values derived from constexpr N are also compile-time constants
    TWO_N_MINUS1 = 2 * N - 1          # e.g. 63 for N=32
    N_SQ = N * N                       # e.g. 1024
    N_SQ_PLUS1 = N_SQ + 1             # e.g. 1025
    TOTAL = N_SQ_PLUS1 * N_SQ_PLUS1   # e.g. 1050625
    # V0 = (2N-1)² = max relative index + 1 (CLS-token sentinel values)
    V0 = TWO_N_MINUS1 * TWO_N_MINUS1  # e.g. 3969
    V1 = V0 + 1                        # e.g. 3970
    V2 = V0 + 2                        # e.g. 3971

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    # Decompose flat index into row/col of the output (N²+1)×(N²+1) matrix
    row = offsets // N_SQ_PLUS1   # in [0, N²]
    col = offsets % N_SQ_PLUS1    # in [0, N²]

    # For inner elements: i = row-1, j = col-1 are positions in [0, N²)
    # (For edge elements row=0 or col=0, these are -1 but we mask them out)
    i = row - 1
    j = col - 1

    # Decompose i,j into grid row/col coordinates within the N×N grid
    ri = i // N   # row coordinate of position i
    ci = i % N    # col coordinate of position i
    rj = j // N
    cj = j % N

    # Relative position bias index (valid for inner elements only)
    inner_val = (ri - rj + (N - 1)) * TWO_N_MINUS1 + (ci - cj + (N - 1))

    # Select output value based on position in the output matrix
    is_corner    = (row == 0) & (col == 0)
    is_first_row = (row == 0) & (col != 0)
    is_first_col = (row != 0) & (col == 0)

    val = tl.where(is_corner,    V2,
          tl.where(is_first_row, V0,
          tl.where(is_first_col, V1,
          inner_val)))

    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


@torch.fx.wrap
def beit_dispatch(route):
    """
    Compute the BEiT relative position bias index table.
    Route string selects the grid size: "32", "24", or "14".
    Returns flat int64 tensor of size (N²+1)².
    """
    if route == "32":
        TOTAL = 1025 * 1025   # (32²+1)²
        out = torch.empty(TOTAL, dtype=torch.int64, device="cuda")
        grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
        rel_pos_idx_kernel[grid](out, 32)
    elif route == "24":
        TOTAL = 577 * 577     # (24²+1)²
        out = torch.empty(TOTAL, dtype=torch.int64, device="cuda")
        grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
        rel_pos_idx_kernel[grid](out, 24)
    else:  # "14"
        TOTAL = 197 * 197     # (14²+1)²
        out = torch.empty(TOTAL, dtype=torch.int64, device="cuda")
        grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
        rel_pos_idx_kernel[grid](out, 14)

    return out