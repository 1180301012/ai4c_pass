import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Simple pattern: element-wise multiplication
    """
    tmp_1 = in_0 * in_1
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load y
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute x * y
    out = x * y

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def mul_kernel_wrapper(x, y):
    n_elements = x.numel()
    # Let Triton handle BLOCK_SIZE selection via autotune
    # We need to provide a grid function
    def grid(META):
        return ((n_elements + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)

    out = torch.empty_like(x)

    mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
    )

    return out


def replacement_func():
    return mul_kernel_wrapper