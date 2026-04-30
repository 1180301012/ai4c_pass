import torch
import triton
import triton.language as tl

# Pattern matching function for fused add + dropout2d
# dropout2d with training=False just passes through, so we can fuse add + dropout2d into a single add
def pattern(in_3, in_4):
    """
    Match the pattern: add + dropout2d(training=False)
    Since dropout2d with training=False returns input unchanged,
    this can be fused into a single element-wise addition.
    """
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_3, in_4):
    return (in_3, in_4)

# Autotune configurations for optimal performance on different tensor sizes
# Expanded configurations for better coverage of different element counts
@triton.autotune(
    configs=[
        # Smaller block sizes for better parallelism on smaller tensors
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        # Medium block sizes
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        # Larger block sizes for maximum throughput on large tensors
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for element-wise addition with autotuning.
    """
    # Calculate program ID and offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary check
    mask = offsets < N
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    result = x + y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_wrapper(in_3, in_4):
    """
    Wrapper function to launch the autotuned Triton kernel.
    """
    x = in_3
    y = in_4
    
    # Get number of elements
    N = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Calculate number of programs needed for maximum parallelism
    BLOCK_SIZE = 1024
    num_programs = max(1, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel with autotuning
    fused_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        N=N,
    )
    
    return out

# Replacement function - returns the wrapper function
def replacement_func():
    return fused_add_wrapper