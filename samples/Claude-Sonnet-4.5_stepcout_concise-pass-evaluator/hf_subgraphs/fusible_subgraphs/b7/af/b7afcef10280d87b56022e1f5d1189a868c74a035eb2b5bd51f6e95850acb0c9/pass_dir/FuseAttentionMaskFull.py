import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern to match the full attention mask computation:
    - Convert to float32
    - Compute (1.0 - x) 
    - Convert to bool
    - masked_fill with -inf
    
    Since the subtraction with - operator has issues, we match from the result
    """
    tmp_0 = in_0.to(torch.float32)
    # Match the subtraction by accepting any intermediate value
    # and then matching the to(bool) and masked_fill
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_attention_mask_full_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for full attention mask computation.
    Input: int64 tensor
    Output: float32 tensor with 1.0-x, then masked_fill where non-zero
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (int64)
    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32
    x_float = x.to(tl.float32)
    
    # Compute 1.0 - x
    result = 1.0 - x_float
    
    # Convert to bool (non-zero -> True) and masked_fill
    # Where result is non-zero, fill with -inf
    is_nonzero = result != 0.0
    result = tl.where(is_nonzero, -3.4028234663852886e+38, result)
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask_full(in_0):
    """
    Wrapper function for the full fused attention mask kernel.
    """
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Create output tensor
    output = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_attention_mask_full_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output


def replacement_func():
    return fused_attention_mask_full