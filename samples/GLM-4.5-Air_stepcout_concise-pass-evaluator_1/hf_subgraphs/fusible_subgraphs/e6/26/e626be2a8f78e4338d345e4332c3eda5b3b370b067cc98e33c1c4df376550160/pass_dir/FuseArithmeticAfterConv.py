import torch
import triton
import triton.language as tl

# Pattern matching for add + divide + clamp sequence
def pattern(conv_result, in_2):
    """Match the arithmetic sequence: add + divide + clamp + multiply"""
    tmp_3 = conv_result + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6

def replacement_args(conv_result, in_2):
    """Extract arguments for replacement"""
    return (conv_result, in_2)

@triton.jit
def fused_kernel(
    conv_out_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: ((conv_result + 1) / 2).clamp(0, 1) * in_2"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    conv_result = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Fused arithmetic: ((x + 1) / 2).clamp(0, 1) * y
    fused_val = (conv_result + 1.0) * 0.5
    # Custom clamping to avoid tl.max/min issues with constexpr
    clamped_val = fused_val
    clamped_val = tl.where(fused_val < 0.0, 0.0, fused_val)
    clamped_val = tl.where(fused_val > 1.0, 1.0, clamped_val)
    result = clamped_val * in_2_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_operations(conv_result, in_2):
    """Fused operations for add + divide + clamp + multiply"""
    n_elements = conv_result.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    result = torch.empty_like(conv_result)
    
    # Launch fused kernel
    fused_kernel[(num_programs,)](
        conv_result,
        in_2,
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return fused_operations