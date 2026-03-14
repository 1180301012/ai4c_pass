import torch
import triton
import triton.language as tl


@triton.jit
def fused_norm_div_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused L2 normalization + division in a single kernel.
    Each program (thread block) handles one row.
    """
    # Get the row this program handles
    row_idx = tl.program_id(0)
    if row_idx >= rows:
        return
    
    # Compute offset for this row
    row_offset = row_idx * cols
    cols_range = tl.arange(0, BLOCK_SIZE)
    col_mask = cols_range < cols
    
    # Load row data
    x = tl.load(in_ptr + row_offset + cols_range, mask=col_mask, other=0.0)
    
    # Compute L2 norm for the row: sqrt(sum(x^2))
    norm = tl.sqrt(tl.sum(x * x, axis=0) + 1e-8)
    
    # Normalize: x / norm
    normalized = x / norm
    
    # Store result
    tl.store(out_ptr + row_offset + cols_range, normalized, mask=col_mask)


@torch.fx.wrap
def fused_norm_div(in_1):
    """
    Fused L2 normalization + division using a single Triton kernel.
    """
    rows, cols = in_1.shape
    BLOCK_SIZE = triton.next_power_of_2(cols)
    
    # Allocate output
    out = torch.empty_like(in_1)
    
    # Single kernel: compute norm and normalize
    grid = (rows,)
    fused_norm_div_kernel[grid](
        in_1, out,
        rows, cols,
        BLOCK_SIZE
    )
    
    return out


def pattern(in_0, in_1):
    """
    Pattern to match: the full forward computation.
    Returns both tmp_1 (norm+div on in_1) and tmp_3 (transpose on in_0).
    Must match exactly what's in the model.
    """
    # Part 1: L2 norm + division on in_1
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    # Clean up tmp_0 (this appears in original model)
    tmp_0 = None
    # Part 2: transpose on in_0
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(torch.device('cuda'))
    # Clean up tmp_2 (this appears in original model)
    tmp_2 = None
    # Return tuple (matching original)
    return (tmp_1, tmp_3)


def replacement_args(in_0, in_1):
    # Pass both inputs to the replacement
    return (in_0, in_1)


def replacement_func():
    # Return a function that handles both operations
    def combined_op(in_0, in_1):
        # Use fused_norm_div for the norm+div part
        tmp_1 = fused_norm_div(in_1)
        # Pass through transpose + to(cuda)
        tmp_2 = in_0.t()
        tmp_3 = tmp_2.cuda()
        return tmp_1, tmp_3
    return combined_op