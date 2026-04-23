import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit

def fused_silu_add_kernel(
    x0_ptr,
    x1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)

    x0_f32 = x0.to(tl.float32)
    x1_f32 = x1.to(tl.float32)
    silu_x1 = x1_f32 * tl.sigmoid(x1_f32)
    out = silu_x1 + x0_f32

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_add(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    fused_silu_add_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_silu_add