import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    # Pattern matches dropout with training=False
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    return tmp_2

def replacement_args(tmp_1):
    return (tmp_1,)

@triton.jit
def dropout_bypass_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel that bypasses dropout when training=False"""
    pid = tl.program_id(0)
    
    # Each program processes a contiguous block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Just copy input to output (no dropout when training=False)
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_values, mask=mask)

@torch.fx.wrap
def dropout_bypass(tmp_1):
    """Bypass dropout operation when training=False - just return input directly"""
    # When training=False, dropout is just an identity operation
    # Return input directly to avoid any overhead
    return tmp_1

def replacement_func():
    return dropout_bypass