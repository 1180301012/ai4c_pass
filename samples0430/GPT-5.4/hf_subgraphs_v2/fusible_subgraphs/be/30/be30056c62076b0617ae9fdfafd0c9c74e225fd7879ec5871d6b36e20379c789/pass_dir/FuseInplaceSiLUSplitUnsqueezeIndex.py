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
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y = x_f32 * tl.sigmoid(x_f32)
    tl.store(x_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def _triton_inplace_silu_split_unsqueeze_index(in_0, in_1):
    n_elements = in_1.numel()
    if n_elements != 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _silu_inplace_kernel[grid](
            x_ptr=in_1,
            n_elements=n_elements,
        )

    tmp_7 = in_0[(None, None, slice(None, None, None))]
    tmp_3 = in_1[:, :, :512]
    tmp_4 = in_1[:, :, 512:1024]
    tmp_6 = in_1[:, :, 1024:1152].unsqueeze(2)
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_func():
    return _triton_inplace_silu_split_unsqueeze_index