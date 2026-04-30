import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Minimal pattern - just a multiply operation."""
    return in_0 * in_1


def replacement_args(in_0, in_1):
    """Extract arguments."""
    return (in_0, in_1)


@triton.jit
def triton_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple element-wise multiply kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_mul(x, y):
    """Triton wrapper for element-wise multiply."""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    triton_mul_kernel[(num_programs,)](
        x, y, out, N, BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return triton_mul