import torch
import triton
import triton.language as tl


def pattern(in_4, tmp_3):
    """
    Pattern: layer_norm output + queries_embeddings addition
    Computes: tmp_3 + in_4
    Returns the addition result (tmp_4)
    Note: tmp_3 is preserved as it's used in the return statement
    """
    tmp_4 = tmp_3 + in_4
    return tmp_4


def replacement_args(in_4, tmp_3):
    return (in_4, tmp_3)


@triton.jit
def add_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise addition kernel for two tensors
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute addition
    result = x + y
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def add_kernel_wrapper(x, y):
    """
    Wrapper for the Triton addition kernel
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    add_kernel[(num_programs,)](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return add_kernel_wrapper