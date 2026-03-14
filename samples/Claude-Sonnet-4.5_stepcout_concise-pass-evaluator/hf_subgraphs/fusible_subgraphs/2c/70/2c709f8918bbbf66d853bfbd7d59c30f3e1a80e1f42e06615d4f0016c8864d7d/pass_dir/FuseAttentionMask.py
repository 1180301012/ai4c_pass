import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_mask_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses: to(float32) -> (1.0 - x) -> (* -FLT_MAX)
    Optimized for memory-bound workloads.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load, convert, compute in single pass
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    
    # Fused computation
    result = (1.0 - x) * -3.4028234663852886e+38
    
    # Store
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask(input_tensor):
    """
    Optimized wrapper for memory-bound fused operations.
    """
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    n_elements = input_tensor.numel()
    
    # Optimal block size from testing
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_mask_kernel[(num_blocks,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_1):
    """
    Pattern to match: type conversion + subtraction + multiplication
    """
    tmp_1 = in_1.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3


def replacement_args(in_1):
    """
    Extract arguments for replacement.
    """
    return (in_1,)


def replacement_func():
    """
    Return the optimized implementation.
    """
    return fused_attention_mask