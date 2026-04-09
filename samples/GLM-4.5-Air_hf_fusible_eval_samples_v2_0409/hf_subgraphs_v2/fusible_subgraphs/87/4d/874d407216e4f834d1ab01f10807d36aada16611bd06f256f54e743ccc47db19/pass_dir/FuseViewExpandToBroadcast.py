import torch
import triton
import triton.language as tl

def pattern(in_tensor, to_expand_with):
    """
    Pattern: view(-1, 1) followed by expand_as(other_tensor)
    This matches where we reshape a 1D tensor to 2D and then expand it to match another tensor's shape
    """
    tmp = in_tensor.view((-1, 1))
    expanded = tmp.expand_as(to_expand_with)
    return expanded

def replacement_args(in_tensor, to_expand_with):
    return (in_tensor, to_expand_with)

@triton.jit
def optimized_expand_kernel(
    in_ptr,
    target_shape_ptr,
    out_ptr,
    in_elements,
    target_rows,
    target_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0) * BLOCK_SIZE
    row_mask = row_idx < target_rows
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < target_cols
    
    # Load input value (all elements in a row should be the same)
    in_val = tl.load(in_ptr + (row_idx % in_elements), mask=(row_idx % in_elements) == 0)
    
    # Store expanded row
    for col in tl.static_range(BLOCK_SIZE):
        if row_mask and col_mask[col]:
            out_idx = row_idx * target_cols + col
            tl.store(out_ptr + out_idx, in_val)

@torch.fx.wrap
def optimized_expand(in_tensor, target_shape):
    """
    Directly expand a tensor to target shape without intermediate view/expand operations
    """
    target_rows, target_cols = target_shape
    in_elements = in_tensor.numel()
    
    # Calculate optimal block size
    BLOCK_SIZE = 128
    num_rows = (target_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((target_rows, target_cols), dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Launch kernel
    optimized_expand_kernel[(num_rows,)](
        in_tensor,
        target_shape,
        out,
        in_elements,
        target_rows,
        target_cols,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    def optimized_func(in_tensor, to_expand_with):
        target_shape = to_expand_with.shape
        return optimized_expand(in_tensor, target_shape)
    return optimized_func