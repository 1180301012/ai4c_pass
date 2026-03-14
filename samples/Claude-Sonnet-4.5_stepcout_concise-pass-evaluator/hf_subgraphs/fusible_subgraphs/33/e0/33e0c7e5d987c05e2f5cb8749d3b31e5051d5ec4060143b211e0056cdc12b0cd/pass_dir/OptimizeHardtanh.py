import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match hardtanh operation that clamps values between 0.0 and 6.0
    """
    result = torch.nn.functional.hardtanh(x, 0.0, 6.0, False)
    return result


def replacement_args(x):
    return (x,)


@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized hardtanh kernel with vectorized memory access
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh: clamp between 0.0 and 6.0
    result = tl.minimum(tl.maximum(x, 0.0), 6.0)
    
    # Vectorized store
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_hardtanh(x):
    """
    Wrapper for optimized hardtanh
    """
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 8192  # Even larger block size for better amortization
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    hardtanh_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_hardtanh