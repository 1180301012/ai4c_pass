import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Fused relu + sigmoid: sigmoid(relu(x))
    # When x <= 0: relu(x) = 0, sigmoid(0) = 0.5
    # When x > 0: relu(x) = x, sigmoid(x) = 1/(1+exp(-x))
    # Combined: 1/(1+exp(-max(0,x))) which correctly gives 0.5 for x<=0
    relu_x = tl.where(x > 0, x, 0.0)
    out = 1.0 / (1.0 + tl.exp(-relu_x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_sigmoid_kernel_autotuned(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    relu_x = tl.where(x > 0, x, 0.0)
    out = 1.0 / (1.0 + tl.exp(-relu_x))
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(x):
    N = x.numel()
    out = torch.empty_like(x)
    # For small tensors, use a single block; for larger ones, use autotuned kernel
    if N <= 2048:
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        relu_sigmoid_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        num_programs = (N + 2048 - 1) // 2048  # approximate, autotune will override
        relu_sigmoid_kernel_autotuned[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
        )
    return out


def replacement_func():
    return fused_relu_sigmoid