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
    key=['left_cols', 'right_cols'],
)
@triton.jit
def _cat2d_i64_kernel(
    left_ptr,
    right_ptr,
    out_ptr,
    left_cols,
    right_cols,
    out_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    left_mask = offs < left_cols
    left_vals = tl.load(left_ptr + row * left_cols + offs, mask=left_mask)
    tl.store(out_ptr + row * out_cols + offs, left_vals, mask=left_mask)

    right_mask = offs < right_cols
    right_vals = tl.load(right_ptr + row * right_cols + offs, mask=right_mask)
    tl.store(out_ptr + row * out_cols + left_cols + offs, right_vals, mask=right_mask)


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
def _ones_f32_kernel(
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
def _runtime_cat2d_i64(left, right):
    left_cols = left.shape[1]
    right_cols = right.shape[1]
    out_cols = left_cols + right_cols
    out = torch.empty((2, out_cols), dtype=left.dtype, device=left.device)
    grid = lambda meta: (2, triton.cdiv(out_cols, meta['BLOCK']))
    _cat2d_i64_kernel[grid](left, right, out, left_cols, right_cols, out_cols)
    return out


@torch.fx.wrap
def _runtime_ones_f32(n):
    out = torch.empty((n,), dtype=torch.float32, device=torch.device(type='cuda'))
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _ones_f32_kernel[grid](out, n)
    return out


@torch.fx.wrap
def dispatch_shared(*args):
    route = args[-1]
    if route == 'cat2d_i64':
        return _runtime_cat2d_i64(args[0], args[1])
    if route == 'ones_f32':
        return _runtime_ones_f32(args[0])
    raise RuntimeError(f'Unknown route: {route}')