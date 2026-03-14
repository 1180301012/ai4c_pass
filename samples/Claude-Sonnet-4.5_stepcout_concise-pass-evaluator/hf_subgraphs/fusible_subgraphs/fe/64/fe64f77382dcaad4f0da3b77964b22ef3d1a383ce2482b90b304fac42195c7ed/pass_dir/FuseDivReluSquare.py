import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern matching for: just square operation
    """
    tmp_2 = torch.square(in_0)
    return tmp_2


def replacement_args(in_0):
    """
    Extract arguments for the replacement kernel
    """
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'VECTOR_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 4096, 'VECTOR_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 8192, 'VECTOR_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'VECTOR_SIZE': 8}),
        triton.Config({'BLOCK_SIZE': 4096, 'VECTOR_SIZE': 8}),
    ],
    key=['n_elements'],
)
@triton.jit
def square_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """
    Optimized kernel for: square operation with vectorization
    """
    # Calculate block offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Square
    out = x * x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_square(in_0):
    """
    Wrapper function to launch the square kernel
    """
    # Calculate total elements
    n_elements = in_0.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_0)
    
    # Define grid - we'll use the autotuned BLOCK_SIZE
    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    # Launch kernel
    square_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


def replacement_func():
    """
    Return the replacement function
    """
    return optimized_square