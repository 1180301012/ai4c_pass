import torch
import triton
import triton.language as tl


@triton.jit
def _scale_flat_kernel(
    weight_ptr,
    features_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * D
    mask = offs < total
    row = offs // D
    w = tl.load(weight_ptr + row, mask=mask, other=0.0)
    feat = tl.load(features_ptr + offs, mask=mask, other=0.0)
    out = w * feat
    tl.store(out_ptr + offs, out, mask=mask)


# Minimal single-return pattern: fuse in_1.view(-1,1) * in_2 only
# Leaves view/expand/new_zeros untouched in the graph (they're cheap/no-copy)
def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@torch.fx.wrap
def _scale_replace(in_1, in_2):
    N = int(in_2.shape[0])
    D = int(in_2.shape[1])
    total = N * D
    BLOCK_SIZE = 1024
    out_scale = torch.empty_like(in_2)
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _scale_flat_kernel[grid](in_1, in_2, out_scale, N, D, BLOCK_SIZE=BLOCK_SIZE)
    return out_scale


def replacement_func():
    return _scale_replace