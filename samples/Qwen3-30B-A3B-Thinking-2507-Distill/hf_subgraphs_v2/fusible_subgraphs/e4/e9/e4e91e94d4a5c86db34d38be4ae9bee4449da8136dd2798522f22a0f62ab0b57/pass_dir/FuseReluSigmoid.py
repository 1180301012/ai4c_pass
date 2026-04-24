import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Upcast to fp32 for numerical stability, then cast back
    x_f32 = x.to(tl.float32)
    x_f32 = tl.sigmoid(x_f32)
    x_out = x_f32.to(x.dtype)

    tl.store(out_ptr + offsets, x_out, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(in_0):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    # Use BLOCK_SIZE=512 so N=512 aligns perfectly (1 block, no waste)
    BLOCK_SIZE = 512
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _sigmoid_kernel[grid](
        in_0,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_relu_sigmoid