import torch
import triton
import triton.language as tl


# Variant: torch.nn.functional.relu(in_0)  [no inplace] + torch.sigmoid
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_relu_sigmoid_v4_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid_v4(in_0):
    n = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_relu_sigmoid_v4_kernel[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return fused_relu_sigmoid_v4