import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['gathered_cols', 'loop_cols'],
)
@triton.jit
def _cat_copy_kernel(
    gathered_ptr,
    loop_ptr,
    out_cat_ptr,
    gathered_cols,
    loop_cols,
    total_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    mask_g = offs < gathered_cols
    g_vals = tl.load(gathered_ptr + row * gathered_cols + offs, mask=mask_g)
    tl.store(out_cat_ptr + row * total_cols + offs, g_vals, mask=mask_g)

    mask_l = offs < loop_cols
    l_vals = tl.load(loop_ptr + row * loop_cols + offs, mask=mask_l)
    tl.store(out_cat_ptr + row * total_cols + gathered_cols + offs, l_vals, mask=mask_l)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['n'],
)
@triton.jit
def _fill_ones_kernel(
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.full([BLOCK], 1.0, tl.float32)
    tl.store(out_ptr + offs, vals, mask=mask)


@torch.fx.wrap
def _runtime_cat_ones(gathered, loop_index, ones_bias):
    gathered_cols = gathered.shape[1]
    loop_cols = loop_index.shape[1]
    total_cols = gathered_cols + loop_cols
    ones_len = ones_bias + gathered_cols

    out_cat = torch.empty((2, total_cols), dtype=gathered.dtype, device=gathered.device)
    out_ones = torch.empty((ones_len,), dtype=torch.float32, device=loop_index.device)

    grid_cat = lambda meta: (2, triton.cdiv(total_cols, meta['BLOCK']))
    _cat_copy_kernel[grid_cat](
        gathered,
        loop_index,
        out_cat,
        gathered_cols,
        loop_cols,
        total_cols,
    )

    grid_ones = lambda meta: (triton.cdiv(ones_len, meta['BLOCK']),)
    _fill_ones_kernel[grid_ones](out_ones, ones_len)
    return out_cat, out_ones


def dispatch_cat_ones(gathered, loop_index, ones_bias):
    return _runtime_cat_ones(gathered, loop_index, ones_bias)