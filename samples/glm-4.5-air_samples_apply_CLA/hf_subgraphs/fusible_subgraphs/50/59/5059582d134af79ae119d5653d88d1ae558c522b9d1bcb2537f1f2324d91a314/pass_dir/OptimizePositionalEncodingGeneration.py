import torch
import triton
import triton.language as tl
from torch import device

def pattern(freqs, in_0):
    # Match the entire positional encoding generation sequence:
    # tmp_1 = torch.arange(N, device=device(type='cuda', index=0))
    # tmp_2 = tmp_1.type_as(tmp_0)  
    # tmp_3 = torch.outer(tmp_2, tmp_0)
    # tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    range_tensor = torch.arange(freqs.shape[0], device=device(type='cuda', index=0))
    typed_range = range_tensor.type_as(in_0)
    outer_result = torch.outer(typed_range, in_0)
    concat_result = torch.cat((outer_result, outer_result), dim=-1)
    return concat_result

def replacement_args(freqs, in_0):
    return (freqs, in_0)

@triton.jit
def positional_encoding_kernel(
    freqs_ptr,
    in_0_ptr,
    out_ptr,
    freqs_size,
    in_0_size,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # 2D grid for outer product computation
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Compute offsets for this program
    x_offset = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_offset = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Create masks for bounds checking
    x_mask = x_offset < freqs_size
    y_mask = y_offset < in_0_size
    
    # Load input data
    freqs = tl.load(freqs_ptr + x_offset, mask=x_mask, other=0.0)
    in_0 = tl.load(in_0_ptr + y_offset, mask=y_mask, other=0.0)
    
    # Compute outer product and concatenate directly
    # Since we want [freqs_size, in_0_size * 2]
    for i in range(BLOCK_SIZE_X):
        for j in range(BLOCK_SIZE_Y):
            if x_mask[i] and y_mask[j]:
                # First half of concatenated result
                offset_first = (y_offset[j] * freqs_size + x_offset[i]).to(tl.int32)
                tl.store(out_ptr + offset_first, freqs[i] * in_0[j], mask=True)
                
                # Second half (same as first half)
                offset_second = ((y_offset[j] + in_0_size) * freqs_size + x_offset[i]).to(tl.int32)
                tl.store(out_ptr + offset_second, freqs[i] * in_0[j], mask=True)

@torch.fx.wrap
def optimized_positional_encoding(freqs, in_0):
    freqs_size = freqs.shape[0]
    in_0_size = in_0.shape[0]
    
    # Output size: [freqs_size, in_0_size * 2]
    output_size = (freqs_size, in_0_size * 2)
    out = torch.empty(output_size, dtype=in_0.dtype, device=freqs.device)
    
    # Set optimal block sizes
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32
    
    # Calculate grid dimensions
    grid_x = (freqs_size + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (in_0_size + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Launch kernel
    positional_encoding_kernel[(grid_x, grid_y)](
        freqs_ptr=freqs,
        in_0_ptr=in_0,
        out_ptr=out,
        freqs_size=freqs_size,
        in_0_size=in_0_size,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out

def replacement_func():
    return optimized_positional_encoding