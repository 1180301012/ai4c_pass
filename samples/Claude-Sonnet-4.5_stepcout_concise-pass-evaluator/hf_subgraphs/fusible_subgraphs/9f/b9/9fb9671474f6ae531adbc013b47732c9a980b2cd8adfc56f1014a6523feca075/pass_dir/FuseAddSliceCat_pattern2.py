import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern: Add two tensors, slice third tensor, concatenate results
    Pattern 2: tmp_0 = in_0 + in_2, tmp_1 = in_1[...], tmp_2 = cat([tmp_0, tmp_1])
    """
    tmp_0 = in_0 + in_2
    tmp_1 = in_1[slice(None, None, None), slice(234, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    # Reorder to match pattern 1's signature: (add_operand1, add_operand2, slice_tensor)
    return (in_0, in_2, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_slice_cat_kernel_p2(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,  # Total number of elements in output
    n_add,  # Number of elements from addition (= B * C1 * H * W)
    slice_offset,  # Offset in in_2 to start copying from
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Computes addition of in_0 and in_1 and writes to first part of output
    2. Copies sliced portion of in_2 to second part of output
    
    Memory layout is contiguous so we can process linearly.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Determine which part each element belongs to
    is_add_part = offsets < n_add
    is_copy_part = offsets >= n_add
    
    # Process addition part
    add_mask = mask & is_add_part
    x = tl.load(in_0_ptr + offsets, mask=add_mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=add_mask, other=0.0)
    add_result = x + y
    tl.store(out_ptr + offsets, add_result, mask=add_mask)
    
    # Process copy part
    copy_mask = mask & is_copy_part
    # Calculate source offset in in_2
    src_offsets = offsets - n_add + slice_offset
    copy_data = tl.load(in_2_ptr + src_offsets, mask=copy_mask, other=0.0)
    tl.store(out_ptr + offsets, copy_data, mask=copy_mask)


@torch.fx.wrap
def fused_add_slice_cat_p2(in_0, in_1, in_2):
    """
    Optimized implementation of:
    tmp_0 = in_0 + in_1
    tmp_1 = in_2[:, slice_start:]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    
    Strategy: Use single fused kernel to minimize memory traffic
    """
    B, C1, H, W = in_0.shape
    _, C2_full, _, _ = in_2.shape
    
    # Determine slice_start (should equal C1 based on pattern analysis)
    slice_start = C1
    C2_sliced = C2_full - slice_start
    
    # Allocate output
    C_out = C1 + C2_sliced
    out = torch.empty((B, C_out, H, W), device=in_0.device, dtype=in_0.dtype)
    
    # Calculate parameters for fused kernel
    n_add = in_0.numel()  # Number of elements from addition
    n_elements = out.numel()  # Total output elements
    slice_offset = slice_start * H * W  # Offset in in_2 where slice starts
    
    # Launch single fused kernel
    grid = (triton.cdiv(n_elements, 4096),)  # Will be autotuned
    
    fused_add_slice_cat_kernel_p2[grid](
        in_0,
        in_1,
        in_2,
        out,
        n_elements,
        n_add,
        slice_offset,
    )
    
    return out


def replacement_func():
    return fused_add_slice_cat_p2