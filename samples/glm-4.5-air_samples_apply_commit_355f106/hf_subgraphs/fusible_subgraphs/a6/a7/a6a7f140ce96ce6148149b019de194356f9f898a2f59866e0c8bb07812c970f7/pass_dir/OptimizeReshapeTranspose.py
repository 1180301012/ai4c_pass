import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the reshape + transpose pattern
    # x is the input tensor [1, 133, 133, 96]
    tmp_5 = x.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    return tmp_6

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_reshape_transpose_kernel(
    in_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (1 * 19 * 19 * 7 * 7 * 96)
    
    # Compute coordinates in original tensor [1, 133, 133, 96]
    idx = offsets
    d0 = idx // (19 * 19 * 7 * 7 * 96)  # batch=1
    
    remaining = idx % (19 * 19 * 7 * 7 * 96)
    d1 = remaining // (19 * 7 * 7 * 96)
    
    remaining = remaining % (19 * 7 * 7 * 96)
    d2 = remaining // (7 * 7 * 96)
    
    remaining = remaining % (7 * 7 * 96)
    d3 = remaining // (7 * 96)
    
    remaining = remaining % (7 * 96)
    d4 = remaining // 96
    
    d5 = remaining % 96
    
    # For input [1, 133, 133, 96] -> reshape to [1, 19, 7, 19, 7, 96], then transpose(2,3)
    # which gives [1, 19, 19, 7, 7, 96]
    
    # Calculate equivalent indices
    # [1, 19, 7, 19, 7, 96] layout:
    # d0, d1, d2_original, d3_original, d4, d5
    # where d2_original = d4, d3_original = d2 = row-major in [19, 19]
    
    d2_original = d4  # From inner 7x7 block
    d3_original = d2  # From outer 19x19 block, transposed
    
    # Transpose dimensions 2 and 3
    d2_final = d3_original
    d3_final = d2_original
    
    # Output index [d0, d1, d2_final, d3_final, d4, d5]
    out_idx = ((d0 * 19 + d1) * 19 + d2_final) * (7 * 7 * 96) + (d3_final * 7 + d4) * 96 + d5
    
    # Read from flat input [1, 133, 133, 96]
    in_idx = d0 * (19 * 7 * 19 * 7 * 96) + d1 * (7 * 19 * 7 * 96) + d2 * (19 * 7 * 96) + d3 * (7 * 96) + d4 * 96 + d5
    
    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + out_idx, val, mask=mask)

@torch.fx.wrap
def optimized_reshape_transpose(in_tensor):
    # Input shape: [1, 133, 133, 96], reshape to [1, 19, 7, 19, 7, 96], transpose (2,3)
    total_elements = 1 * 19 * 19 * 7 * 7 * 96
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((1, 19, 19, 7, 7, 96), dtype=in_tensor.dtype, device=in_tensor.device)
    
    optimized_reshape_transpose_kernel[(num_programs,)](
        in_tensor,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_reshape_transpose