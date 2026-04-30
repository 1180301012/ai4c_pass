import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU using erf approximation: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Cast to float32 for erf which doesn't support fp16/bf16, then cast back
    x_fp32 = x.to(tl.float32)
    gelu_fp32 = x_fp32 * 0.5 * (1.0 + tl.math.erf(x_fp32 * 0.7071067811865476))
    gelu_out = gelu_fp32.to(x.type)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


@torch.fx.wrap
def gelu_kernel_wrapper(x):
    """
    GELU activation kernel.
    """
    n_elements = x.numel()
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Configure grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(x):
    """
    Match gelu pattern.
    """
    return torch.nn.functional.gelu(x, approximate='none')


def replacement_args(x):
    return (x,)


def replacement_func():
    return gelu_kernel_wrapper