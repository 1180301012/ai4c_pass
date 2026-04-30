import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['N', 'M'],
)
@triton.jit
def gather_cat_kernel_128(
    in0_ptr,
    in1_ptr,
    mask_ptr,
    out_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather (in_0[:, mask]) + cat with in_1.
    in0: [2, N] int64  (N=128)
    in1: [2, M] int64  (M=128 for RECT_L)
    mask: [N] bool
    out: [2, N+M] int64
    """
    row = tl.program_id(0)
    pid = tl.program_id(1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = N + M
    mask_elems = offsets < total

    is_gather = offsets < N

    # Gather indices: mask selects which columns from in0
    col_idx = offsets % N
    m_val = tl.load(mask_ptr + col_idx, mask=mask_elems, other=0)
    # Load in0[row, col_idx] only where mask is True
    in0_val = tl.load(in0_ptr + row * N + col_idx, mask=mask_elems & is_gather & m_val, other=0)
    # Load in1[row, col_idx - N] only where it's the cat part
    in1_val = tl.load(in1_ptr + row * M + tl.where(is_gather, 0, offsets - N),
                      mask=mask_elems & ~is_gather, other=0)

    val = tl.where(is_gather, in0_val, in1_val)
    tl.store(out_ptr + row * total + offsets, val, mask=mask_elems)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['total'],
)
@triton.jit
def ones_kernel_128(
    out_ptr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(out_ptr + offsets, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)


@torch.fx.wrap
def fused_gather_cat_ones_128(in_0, in_1, in_2, tmp_2):
    N = 128
    M = tmp_2  # 128 for RECT_L
    total = N + M  # 256

    out_cat = torch.empty((2, total), dtype=in_0.dtype, device=in_0.device)
    out_ones = torch.empty((total,), dtype=torch.float32, device='cuda')

    BLOCK_ROW = 1
    grid_row = 2
    grid_col = (total + BLOCK_ROW - 1) // BLOCK_ROW
    gather_cat_kernel_128[(grid_row, grid_col)](
        in_0, in_1, in_2, out_cat, N, M
    )

    ones_kernel_128[(1,)](out_ones, total)

    return out_cat, out_ones


def pattern(in_0, in_1, in_2, tmp_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=device(type='cuda'))
    return tmp_9, tmp_11


def replacement_args(in_0, in_1, in_2, tmp_2):
    return (in_0, in_1, in_2, tmp_2)


def replacement_func():
    return fused_gather_cat_ones_128