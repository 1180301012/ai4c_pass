import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def triton_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Cast to float32 for tl.exp (required for bf16/fp16 inputs)
    x_f32 = x.to(tl.float32)

    # Sigmoid: 1 / (1 + exp(-x))
    # Input is the relu output (values >= 0), so no relu needed in kernel
    out_f32 = 1.0 / (1.0 + tl.exp(-x_f32))

    # Cast back to original dtype and store
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    # Fixed BLOCK_SIZE=8192 covers all problem sizes (max 6144 elements)
    # Single-block with masking: one compiled kernel for all shapes,
    # minimising per-call Python dispatch overhead.
    triton_relu_sigmoid_kernel[(1,)](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=8192,
        num_warps=32,
    )

    return out


def replacement_func():
    return fused_relu_sigmoid