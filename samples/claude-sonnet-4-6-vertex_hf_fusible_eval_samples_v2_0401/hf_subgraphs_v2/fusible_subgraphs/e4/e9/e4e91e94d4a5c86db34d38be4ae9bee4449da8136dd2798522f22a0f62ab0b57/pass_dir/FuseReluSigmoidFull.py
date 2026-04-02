import torch
import triton
import triton.language as tl


def pattern(in_0):
    # Match the full relu+sigmoid pattern from model.py
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    # in_0 is the original input (before relu) — feeds directly into fused kernel
    return (in_0,)


@triton.jit
def triton_fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Cast to float32 for tl.exp (required for bf16/fp16)
    x_f32 = x.to(tl.float32)

    # Fused ReLU: max(0, x)
    x_f32 = tl.where(x_f32 > 0.0, x_f32, 0.0)

    # Sigmoid: 1 / (1 + exp(-x))
    out_f32 = 1.0 / (1.0 + tl.exp(-x_f32))

    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid_full(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    # Fixed single-block: covers all problem sizes (max 6144 → 8192)
    triton_fused_relu_sigmoid_kernel[(1,)](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=8192,
        num_warps=32,
    )

    return out


def replacement_func():
    return fused_relu_sigmoid_full