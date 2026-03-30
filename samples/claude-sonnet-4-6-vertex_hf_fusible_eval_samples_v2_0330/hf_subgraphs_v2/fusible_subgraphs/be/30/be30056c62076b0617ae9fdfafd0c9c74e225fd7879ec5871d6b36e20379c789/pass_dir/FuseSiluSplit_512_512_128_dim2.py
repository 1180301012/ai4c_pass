import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    x_sigmoid = tl.sigmoid(x)
    out = x * x_sigmoid
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_split_unsqueeze(in_0, in_1):
    n_elements = in_1.numel()
    out = torch.empty_like(in_1)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](in_1, out, n_elements)

    # Split along dim 2 — these are views into `out` (no data copy)
    tmp_3 = out[:, :, :512]
    tmp_4 = out[:, :, 512:1024]
    tmp_5 = out[:, :, 1024:]

    # unsqueeze dim 2 on the 128-chunk (view operation)
    tmp_6 = tmp_5.unsqueeze(2)

    # Add two leading unit dimensions to in_0 (view operation)
    tmp_7 = in_0[(None, None, slice(None, None, None))]

    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_func():
    return silu_split_unsqueeze