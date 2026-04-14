import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 4}),
        triton.Config({'BLOCK_N': 8}),
        triton.Config({'BLOCK_N': 16}),
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
    ],
    key=['N'],
)
@triton.jit
def _scale_d128_kernel(
    weight_ptr,
    features_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    d_offs = tl.arange(0, 128)
    flat = n_offs[:, None] * 128 + d_offs[None, :]
    mask_2d = n_mask[:, None]

    w = tl.load(weight_ptr + n_offs, mask=n_mask, other=0.0)
    feat = tl.load(features_ptr + flat, mask=mask_2d, other=0.0)
    out = w[:, None] * feat
    tl.store(out_ptr + flat, out, mask=mask_2d)


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((128, 128))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _fused_scale_expand_128_128(in_0, in_1, in_2):
    N = in_1.shape[0]
    out_scale = torch.empty_like(in_2)
    grid = lambda meta: ((N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],)
    _scale_d128_kernel[grid](in_1, in_2, out_scale, N)
    out_idx = in_0.view(-1, 1).expand_as(out_scale)
    out_zeros = torch.zeros((128, 128), dtype=in_2.dtype, device=in_2.device)
    return (out_idx, out_zeros, out_scale)


def replacement_func():
    return _fused_scale_expand_128_128