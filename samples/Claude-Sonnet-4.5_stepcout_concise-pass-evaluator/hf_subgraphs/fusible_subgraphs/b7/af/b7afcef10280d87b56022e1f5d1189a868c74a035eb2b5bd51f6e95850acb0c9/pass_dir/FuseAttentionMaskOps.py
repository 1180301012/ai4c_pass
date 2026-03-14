import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match: convert to bool, then masked_fill
    This matches the last 2 operations of the attention mask computation
    """
    tmp_0 = in_0.to(torch.bool)
    tmp_1 = in_0.masked_fill(tmp_0, -3.4028234663852886e+38)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_masked_fill_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for masked_fill with bool conversion
    Input: float32 tensor
    Output: float32 tensor with masked_fill applied
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (float32) with vectorized access
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused: Convert to bool (non-zero -> True) and masked_fill
    # Where x is non-zero, fill with -inf
    is_nonzero = x != 0.0
    result = tl.where(is_nonzero, -3.4028234663852886e+38, x)
    
    # Store output with vectorized access
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_masked_fill(in_0):
    """
    Wrapper function for the fused masked_fill kernel.
    Takes float32 input, produces float32 output.
    """
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Create output tensor
    output = torch.empty_like(in_0, dtype=in_0.dtype)
    
    # Use optimized block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_masked_fill_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_masked_fill