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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    # SiLU: x * sigmoid(x)
    result = x_f32 * tl.sigmoid(x_f32)

    tl.store(x_ptr + offsets, result.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    n_elements = in_1.numel()

    _silu_inplace_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
        in_1,
        n_elements,
    )

    # All view ops — zero allocation, zero copy
    tmp_3 = in_1[:, :, 0:512]
    tmp_4 = in_1[:, :, 512:1024]
    tmp_5 = in_1[:, :, 1024:1152]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[None, None, :]

    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_func():
    return fused_silu_split_unsqueeze