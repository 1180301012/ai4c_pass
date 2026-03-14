import torch
import triton
import triton.language as tl

# Pattern matching for the arithmetic sequence: add + divide + clamp
def pattern(conv_result):
    """Match the arithmetic sequence: add + divide + clamp"""
    tmp_3 = conv_result + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    return tmp_5

def replacement_args(conv_result):
    """Extract arguments for replacement"""
    return (conv_result,)

@triton.jit
def arithmetic_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized arithmetic kernel: (x + 1) / 2 clamped to [0, 1]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused arithmetic: (x + 1) / 2 with clamping
    fused_val = (x + 1.0) * 0.5
    # Use conditional expressions for clamping
    clamped_val = fused_val
    clamped_val = tl.maximum(clamped_val, 0.0)
    clamped_val = tl.minimum(clamped_val, 1.0)
    
    # Store result
    tl.store(out_ptr + offsets, clamped_val, mask=mask)

@torch.fx.wrap

def optimized_arithmetic(conv_result):
    """Optimized arithmetic operations using Triton with autotuning"""
    n_elements = conv_result.numel()
    
    # Use dynamic block sizing based on tensor size
    if n_elements < 8192:
        BLOCK_SIZE = 128
    elif n_elements < 65536:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    result = torch.empty_like(conv_result)
    
    # Launch arithmetic kernel
    arithmetic_kernel[(num_programs,)](
        conv_result,
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return optimized_arithmetic