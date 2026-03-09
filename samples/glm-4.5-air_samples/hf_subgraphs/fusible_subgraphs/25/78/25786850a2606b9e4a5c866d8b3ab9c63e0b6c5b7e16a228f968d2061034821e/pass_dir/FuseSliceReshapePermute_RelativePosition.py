import torch
import triton
import triton.language as tl

def pattern(x):
    # tmp_11 = tmp_4[slice(None, 729, None)]
    tmp_11 = x[slice(None, 729, None)]  
    # tmp_12 = tmp_11.reshape(1, 27, 27, -1)
    tmp_12 = tmp_11.reshape(1, 27, 27, -1)
    # tmp_13 = tmp_12.permute(0, 3, 1, 2)
    tmp_13 = tmp_12.permute(0, 3, 1, 2)
    return tmp_13

def replacement_args(x):
    return (x,)

@triton.jit
def slice_reshape_permute_kernel(
    x_ptr,
    out_ptr,
    n_elements_total,
    n_elements_slice,
    slice_start: tl.constexpr,
    reshape_d1: tl.constexpr,
    reshape_d2: tl.constexpr,
    reshape_d3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_slice
    
    # Load the sliced data (first 729 elements)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Simplified: reshape [729] -> [27, 27, 12] directly
    # d1 = element_idx // (27 * 12) = element_idx // 324
    d1 = offsets // (reshape_d2 * reshape_d3)
    # d2 = (element_idx % 324) // 12
    d2 = (offsets % (reshape_d2 * reshape_d3)) // reshape_d3  
    # d3 = element_idx % 12
    d3 = offsets % reshape_d3
    
    # Calculate output index: [27, 27, 12] -> [12, 27, 27] permutation
    # New order: d3 -> first dim, d1 -> second dim, d2 -> third dim
    output_idx = d3 * (reshape_d1 * reshape_d2) + d1 * reshape_d2 + d2
    
    # Store result
    tl.store(out_ptr + output_idx, x, mask=mask)

@torch.fx.wrap
def optimized_slice_reshape_permute(x):
    # Input: [732, 12], we slice to [729, 12]
    slice_start = 0
    slice_size = 729
    
    # Reshape and permute: [729, 12] -> [1, 27, 27, 12] -> [1, 12, 27, 27]
    reshape_d1, reshape_d2, reshape_d3 = 27, 27, 12
    output_shape = (1, reshape_d3, reshape_d1, reshape_d2)  # [1, 12, 27, 27]
    output_elements = slice_size * reshape_d3  # 729 * 12 = 8748
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (slice_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    slice_reshape_permute_kernel[(num_programs,)](
        x,
        output,
        x.numel(),  # total elements in input
        slice_size,  # elements to process (729)
        slice_start,  # slice start (0)
        27, 27, 12,  # reshape dimensions
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_slice_reshape_permute