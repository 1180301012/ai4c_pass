import torch
import triton
import triton.language as tl


def pattern(in_0, in_3, in_2):
    in_4 = in_3 + in_0
    tmp_0 = in_4 + in_2
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2


def replacement_args(in_0, in_3, in_2):
    return (in_0, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr, in_3_ptr, in_2_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)

    # Fuse: relu(in_0 + in_3 + in_2)
    result = in_0 + in_3 + in_2
    result = tl.where(result > 0, result, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_3, in_2):
    N = in_0.numel()

    out = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    fused_add_add_relu_kernel[grid](
        in_0_ptr=in_0, in_3_ptr=in_3, in_2_ptr=in_2, out_ptr=out,
        n_elements=N,
    )

    return out


def replacement_func():
    return fused_add_add_relu