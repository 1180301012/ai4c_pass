import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Minimal pattern - just optimize masked_fill
    """
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def optimized_masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized masked_fill kernel with vectorization
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with eviction policy hint for better cache usage
    in_val = tl.load(in_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_last')
    
    # Compute in_0 != 0 and masked_fill in one step
    ne_zero = in_val != 0.0
    result = tl.where(ne_zero, -1000.0, in_val)
    
    # Store result with eviction policy hint
    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_last')


@torch.fx.wrap
def optimized_masked_fill(in_0):
    """
    Wrapper function that launches the optimized kernel
    """
    n_elements = in_0.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_0)
    
    # Use larger block size for better performance - tune for the input size
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    optimized_masked_fill_kernel[grid](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_masked_fill