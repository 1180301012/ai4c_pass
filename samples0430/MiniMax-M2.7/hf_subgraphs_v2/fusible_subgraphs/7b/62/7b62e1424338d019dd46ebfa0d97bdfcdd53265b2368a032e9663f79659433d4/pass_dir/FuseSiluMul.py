import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused silu + multiplication kernel.
    
    silu(x) = x * sigmoid(x)
    This kernel computes: silu(x) * y
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute silu: x * sigmoid(x)
    # Use sigmoid(x) = 1 / (1 + exp(-x)) for numerical stability
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    
    # Multiply silu(x) * y
    out = silu_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused silu + multiply operation using Triton."""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    silu_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def silu_mul_wrapper(x, y):
    return silu_mul(x, y)


def pattern(in_0, in_1):
    """Match the silu + mul + dropout pattern.
    
    Since dropout with p=0 is a no-op, we can fuse silu + mul.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return silu_mul_wrapper