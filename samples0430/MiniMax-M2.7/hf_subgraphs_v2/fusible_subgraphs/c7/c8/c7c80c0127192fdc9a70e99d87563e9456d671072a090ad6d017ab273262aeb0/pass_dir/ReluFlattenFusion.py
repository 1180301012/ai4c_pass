import torch
import triton
import triton.language as tl

@triton.jit
def flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Flatten kernel for contiguous data.
    For [N, C, 1, 1] -> [N, C], this is a simple reshape with no data movement.
    We use a kernel anyway to match the original signature.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store result (data is contiguous, so simple copy is fine)
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def flatten_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the flatten operation.
    Input shape: [N, C, 1, 1] -> Output shape: [N, C]
    Uses a Triton kernel to copy data to a new contiguous tensor.
    """
    N, C = x.shape[0], x.shape[1]
    n_elements = N * C
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor using allowed API
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    # Launch kernel to copy data
    flatten_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """
    Match the flatten pattern: tensor.flatten(1, -1)
    This handles the case where a tensor has shape [N, C, 1, 1] and needs to be flattened to [N, C].
    """
    result = x.flatten(1, -1)
    return result

def replacement_args(x):
    return (x,)

def replacement_func():
    return flatten_wrapper