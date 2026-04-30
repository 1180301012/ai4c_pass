import torch
import triton
import triton.language as tl

# Pattern matching - must mirror model.py exactly (without cleanup statements)
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return (tmp_2,)

# Argument extraction
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel: fuses silu(x) * y into one kernel
@triton.jit
def fused_silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # Then multiply by y
    sigmoid_x = tl.sigmoid(x)
    silu_x = x * sigmoid_x
    out = silu_x * y

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    fused_silu_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


def replacement_func():
    return fused_silu_mul