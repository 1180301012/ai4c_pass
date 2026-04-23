import torch
import triton
import triton.language as tl


@triton.jit
def silu_add_kernel(
    x_ptr,
    silu_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Silu + Add kernel.
    Computes: silu(silu_input) + add_input = silu_input * sigmoid(silu_input) + add_input
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    silu_input = tl.load(silu_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_val = tl.sigmoid(silu_input)
    silu_out = silu_input * sigmoid_val
    
    # Fused add
    out = silu_out + x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_add_wrapper(x, silu_input):
    """Wrapper for the fused silu + add operation"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    silu_add_kernel[(num_programs,)](
        x_ptr=x,
        silu_ptr=silu_input,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(x, silu_input):
    """
    Match the pattern: silu(silu_input) + x
    This matches:
        tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_1 = tmp_0 + in_0
    """
    tmp_0 = torch.nn.functional.silu(silu_input, inplace=True)
    tmp_1 = tmp_0 + x
    return tmp_1


def replacement_args(x, silu_input):
    return (x, silu_input)


def replacement_func():
    return silu_add_wrapper