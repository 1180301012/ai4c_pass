import torch
import triton
import triton.language as tl


@triton.jit
def fused_div_add_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: out = x / 8.0 + y + z
    This fuses 3 operations into a single kernel for better performance.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: x/8.0 + y + z
    out = (x / 8.0) + y + z
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_div_add_add(x, y, z):
    """
    Wrapper for the fused kernel.
    Computes: out = x / 8.0 + y + z in a single kernel launch.
    """
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    fused_div_add_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern(in_0, in_1, in_2):
    """
    Pattern: Fused Div-Add-Add operations from attention mechanism.
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2
    tmp_2 = tmp_1 + in_1
    Returns tmp_2 (the final result).
    """
    tmp_0 = in_0 / 8.0
    tmp_0 = tmp_0 + in_2
    tmp_1 = tmp_0
    tmp_2 = tmp_1 + in_1
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_2, in_1)


def replacement_func():
    return fused_div_add_add