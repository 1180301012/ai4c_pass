import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: slice + reshape + permute
    Input x shape: [732, 12]
    After slice: [729, 12]
    After reshape(1, 27, 27, -1): [1, 27, 27, 12]
    After permute(0, 3, 1, 2): [1, 12, 27, 27]
    """
    tmp = x[slice(None, 729, None)]
    tmp2 = tmp.reshape(1, 27, 27, -1)
    result = tmp2.permute(0, 3, 1, 2)
    return result

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def slice_reshape_permute_kernel_1d(
    input_ptr,
    output_ptr,
    in_dim1,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized 1D slice + reshape + permute kernel
    Input: [732, in_dim1]
    After slice: [729, in_dim1]
    After reshape: [1, 27, 27, in_dim1]
    After permute(0, 3, 1, 2): [1, in_dim1, 27, 27]
    """
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Output layout: [1, in_dim1, 27, 27]
    # Convert linear offset to output coordinates
    c_idx = offsets // (27 * 27)
    remainder = offsets % (27 * 27)
    h_idx = remainder // 27
    w_idx = remainder % 27
    
    # Before permute: [1, h, w, c]
    # After reshape from [729, in_dim1]: linear_idx = h * 27 * in_dim1 + w * in_dim1 + c
    # In sliced tensor [729, in_dim1]: row = (h * 27 + w), col = c
    row_idx = h_idx * 27 + w_idx
    
    # Input offset: row * in_dim1 + c
    input_offset = row_idx * in_dim1 + c_idx
    
    # Load and store
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, data, mask=mask)

@torch.fx.wrap
def fused_slice_reshape_permute(x):
    """
    Fused implementation of slice + reshape + permute
    Input: [732, 12]
    Output: [1, 12, 27, 27]
    """
    in_dim0, in_dim1 = x.shape
    slice_end = 729
    
    # First do the slice
    x_sliced = x[:slice_end, :]
    
    # Output shape after all operations
    output = torch.empty((1, in_dim1, 27, 27), dtype=x.dtype, device=x.device)
    
    # Total elements to process
    total_elements = in_dim1 * 27 * 27
    
    # Launch kernel (BLOCK_SIZE will be autotuned)
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    slice_reshape_permute_kernel_1d[grid](
        x_sliced,
        output,
        in_dim1,
        total_elements,
    )
    
    return output

def replacement_func():
    return fused_slice_reshape_permute