import torch
import triton
import triton.language as tl
from pass_dir.pattern_helper import make_silu_add_pattern

pattern = make_silu_add_pattern()


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_silu_add_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)

    # Compute SiLU(x) + y in float32 for numerical accuracy
    x_fp32 = x.to(tl.float32)
    sigmoid_x = tl.sigmoid(x_fp32)
    silu_x = x_fp32 * sigmoid_x
    result = silu_x + y.to(tl.float32)

    # Cast back to input dtype and store
    result = result.to(x.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_silu_add(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_silu_add_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=1024,
    )

    return out


def replacement_func():
    return fused_silu_add