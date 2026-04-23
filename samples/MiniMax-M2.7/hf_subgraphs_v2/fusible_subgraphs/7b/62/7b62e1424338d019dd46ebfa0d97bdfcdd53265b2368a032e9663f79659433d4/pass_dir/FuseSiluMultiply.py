import torch
import triton
import triton.language as tl


@triton.jit
def silu_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU + element-wise multiply kernel.
    Computes: silu(x) * y
    silu(x) = x * sigmoid(x)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid(x)
    sigmoid_x = tl.sigmoid(x)

    # Compute silu(x) = x * sigmoid(x)
    silu_x = x * sigmoid_x

    # Compute result = silu(x) * y
    result = silu_x * y

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def silu_multiply_wrapper(x, y):
    """
    Wrapper for the fused silu + multiply kernel.
    """
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensor with same properties as x
    out = torch.empty_like(x)

    # Launch kernel
    silu_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern(in_0, in_1):
    """
    Match the pattern: silu(in_0) * in_1
    followed by dropout with p=0.0 (no-op)
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    # dropout with p=0.0 is a no-op, just pass through
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return silu_multiply_wrapper