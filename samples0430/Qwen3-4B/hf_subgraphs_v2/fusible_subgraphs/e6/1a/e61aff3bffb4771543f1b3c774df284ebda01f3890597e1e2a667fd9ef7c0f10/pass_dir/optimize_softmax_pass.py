import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(in_2.shape[0], 1, -1)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=2)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_softmax_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute softmax with numerical stability
    max_val = tl.max(x)
    x_exp = tl.exp(x - max_val)
    sum_exp = tl.sum(x_exp)
    softmax = x_exp / sum_exp

    # Store result
    tl.store(y_ptr + offsets, softmax, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    tmp_3 = torch.empty(0, 1, 0, device=conv2d.device, dtype=conv2d.dtype)
    N = tmp_3.numel()
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Initialize output
    out = torch.empty((conv2d.shape[0], 1, tmp_3.shape[2]), device=conv2d.device, dtype=conv2d.dtype)

    # Launch kernel
    optimized_softmax_kernel[(num_blocks,)](
        x_ptr=tmp_3,
        y_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return the unsqueezed output
    return out.unsqueeze(-1)

def replacement_func():
    return kernel_wrapper