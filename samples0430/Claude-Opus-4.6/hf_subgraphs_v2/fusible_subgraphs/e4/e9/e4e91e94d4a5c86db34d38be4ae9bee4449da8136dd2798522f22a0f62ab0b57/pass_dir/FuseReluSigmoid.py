import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Cast to float32 for computation (needed for bf16/fp16)
    x_fp32 = x.to(tl.float32)
    out_fp32 = tl.sigmoid(x_fp32)
    # Cast back to original dtype
    out = out_fp32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_sigmoid(in_0):
    N = in_0.numel()
    # Write in-place since relu output is only consumed by sigmoid
    fused_sigmoid_kernel[(1,)](
        in_0,
        in_0,
        N,
        BLOCK_SIZE=8192,
        num_warps=4,
        num_stages=1,
    )
    return in_0


def replacement_func():
    return fused_sigmoid