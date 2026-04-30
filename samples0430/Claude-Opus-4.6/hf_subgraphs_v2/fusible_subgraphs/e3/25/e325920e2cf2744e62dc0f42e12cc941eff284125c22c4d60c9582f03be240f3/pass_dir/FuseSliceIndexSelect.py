import torch
import triton
import triton.language as tl


def pattern(x, indices):
    return x.index_select(-2, indices)


def replacement_args(x, indices):
    return (x, indices)


@triton.jit
def gather_rows_kernel(
    src_ptr,
    indices_ptr,
    out_ptr,
    M,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    # Load indices and cast to int32 for faster address computation
    indices = tl.load(indices_ptr + row_offsets, mask=row_mask, other=0).to(tl.int32)

    # Gather rows - vectorized across D dimension
    col_offsets = tl.arange(0, D)
    src_offsets = indices[:, None] * D + col_offsets[None, :]
    out_offsets = row_offsets[:, None] * D + col_offsets[None, :]
    mask_2d = row_mask[:, None]

    vals = tl.load(src_ptr + src_offsets, mask=mask_2d, other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask_2d)


@torch.fx.wrap
def fast_index_select(x, indices):
    M = indices.shape[0]
    D = x.shape[1]

    out = torch.empty((M, D), dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)

    gather_rows_kernel[grid](
        x, indices, out,
        M, D,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=1,
    )

    return out


def replacement_func():
    return fast_index_select