import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    in_3 += in_0
    in_4 = in_3
    in_4 += in_2
    tmp_0 = in_4
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return (tmp_2, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit

def _fused_add_add_relu_kernel(x0_ptr, x2_ptr, x3_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    x3 = tl.load(x3_ptr + offsets, mask=mask, other=0.0)

    out = x3 + x0 + x2
    out = tl.maximum(out, 0)

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_add_relu_view_permute(in_0, in_1, in_2, in_3):
    out = torch.empty_like(in_3)
    n_elements = out.numel()

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _fused_add_add_relu_kernel[grid](
        in_0,
        in_2,
        in_3,
        out,
        n_elements,
    )

    tmp_4 = in_1.view(1, 32, -1).permute(0, 2, 1)
    return (out, tmp_4)


def replacement_func():
    return fused_add_add_relu_view_permute