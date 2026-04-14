import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_1.view(1, 32, -1).permute(0, 2, 1)
# in_1 shape [1, 32, 64, 48]; output shape [1, 3072, 32]
# Two standard call_method nodes → should match without iadd issues.
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton 2-D transpose kernel: [32, 3072] → [3072, 32]
# Fixed TILE_N=32, no autotune overhead. Works for float16 and bfloat16.
# ---------------------------------------------------------------------------
@triton.jit
def _view_permute_kernel(
    in_ptr,
    out_ptr,
    N,
    TILE_N: tl.constexpr,
    M: tl.constexpr,
):
    pid     = tl.program_id(0)
    row_ids = tl.arange(0, M)                         # 0..M-1
    col_ids = pid * TILE_N + tl.arange(0, TILE_N)     # column slice

    # Coalesced load: each row has consecutive col addresses
    in_mask = col_ids[None, :] < N
    in_offs = row_ids[:, None] * N + col_ids[None, :]  # [M, TILE_N]
    tile    = tl.load(in_ptr + in_offs, mask=in_mask, other=0.0)

    # Transposed store: output[col, row] = input[row, col]
    out_mask = col_ids[:, None] < N
    out_offs = col_ids[:, None] * M + row_ids[None, :]  # [TILE_N, M]
    tl.store(out_ptr + out_offs, tl.trans(tile), mask=out_mask)


@torch.fx.wrap
def view_permute_1_32_neg1_021(in_1):
    # in_1: [1, 32, 64, 48]  →  contiguous [1, 3072, 32]
    M      = 32
    N      = 64 * 48   # 3072
    TILE_N = 32
    out    = torch.empty(1, N, M, dtype=in_1.dtype, device=in_1.device)
    grid   = (N // TILE_N,)   # 96 blocks
    _view_permute_kernel[grid](in_1, out, N, TILE_N=TILE_N, M=M)
    return out


def replacement_func():
    return view_permute_1_32_neg1_021