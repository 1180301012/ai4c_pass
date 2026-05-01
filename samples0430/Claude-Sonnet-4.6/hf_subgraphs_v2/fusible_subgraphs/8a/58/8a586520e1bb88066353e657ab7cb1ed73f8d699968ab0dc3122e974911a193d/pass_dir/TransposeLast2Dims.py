import torch
import triton
import triton.language as tl


def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 16, 'BLOCK_COLS': 16}),
        triton.Config({'BLOCK_ROWS': 32, 'BLOCK_COLS': 32}),
        triton.Config({'BLOCK_ROWS': 64, 'BLOCK_COLS': 16}),
        triton.Config({'BLOCK_ROWS': 16, 'BLOCK_COLS': 64}),
        triton.Config({'BLOCK_ROWS': 32, 'BLOCK_COLS': 16}),
        triton.Config({'BLOCK_ROWS': 16, 'BLOCK_COLS': 32}),
    ],
    key=['rows', 'cols'],
)
@triton.jit
def _transpose_last2_kernel(
    inp_ptr, out_ptr,
    rows, cols,
    stride_in_bh, stride_out_bh,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # Each program handles a [BLOCK_ROWS, BLOCK_COLS] tile
    pid_bh  = tl.program_id(0)  # batch * head index
    pid_row = tl.program_id(1)  # tile row
    pid_col = tl.program_id(2)  # tile col

    row_offs = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offs = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    # Load: inp[bh, row, col] -> shape [BLOCK_ROWS, BLOCK_COLS]
    mask_in = (row_offs[:, None] < rows) & (col_offs[None, :] < cols)
    inp_offset = pid_bh * stride_in_bh + row_offs[:, None] * cols + col_offs[None, :]
    data = tl.load(inp_ptr + inp_offset, mask=mask_in, other=0.0)

    # Store transposed: out[bh, col, row]
    mask_out = (col_offs[:, None] < cols) & (row_offs[None, :] < rows)
    out_offset = pid_bh * stride_out_bh + col_offs[:, None] * rows + row_offs[None, :]
    tl.store(out_ptr + out_offset, tl.trans(data), mask=mask_out)


@torch.fx.wrap
def triton_transpose_last2(x):
    # x: [..., rows, cols] -> [..., cols, rows]
    rows = x.shape[-2]
    cols = x.shape[-1]
    BH   = x.numel() // (rows * cols)   # product of all leading dims

    # Build output shape with last two dims swapped
    out_shape = list(x.shape)
    out_shape[-2] = cols
    out_shape[-1] = rows
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        BH,
        (rows + meta['BLOCK_ROWS'] - 1) // meta['BLOCK_ROWS'],
        (cols + meta['BLOCK_COLS'] - 1) // meta['BLOCK_COLS'],
    )

    _transpose_last2_kernel[grid](
        x, out,
        rows, cols,
        rows * cols,   # stride_in_bh
        cols * rows,   # stride_out_bh
    )

    return out


def replacement_func():
    return triton_transpose_last2