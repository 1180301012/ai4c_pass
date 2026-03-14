import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: add -> silu
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_silu_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: out = silu(in_0 + in_1)
    """
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load in_0 and in_1
    x0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise addition
    x = x0 + x1
    
    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_silu_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused kernel.
    """
    # Flatten inputs for element-wise operation
    in_0_flat = in_0.flatten()
    in_1_flat = in_1.flatten()
    
    n_elements = in_0_flat.numel()
    
    # Allocate output
    out = torch.empty_like(in_0_flat)
    
    # Define block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_silu_kernel[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return out.reshape(in_0.shape)


def replacement_func():
    return fused_add_silu_wrapper