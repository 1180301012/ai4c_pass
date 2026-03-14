import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the pattern: max(keepdim) -> expand -> subtract -> softmax
    This is a custom softmax variant: softmax(max(x) - x)
    """
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_max_sub_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for computing softmax(max(x, dim=-1, keepdim=True) - x)
    Each program handles one row (along the last dimension).
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Column offsets for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input values for this row
    input_vals = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Find max value in the row
    max_val = tl.max(input_vals, axis=0)
    
    # Compute max - input
    diff = max_val - input_vals
    
    # Apply softmax to diff: exp(diff) / sum(exp(diff))
    exp_vals = tl.exp(diff)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_output = exp_vals / sum_exp
    
    # Store the result
    tl.store(output_ptr + row_start + col_offsets, softmax_output, mask=mask)


@torch.fx.wrap
def fused_max_sub_softmax(input_tensor):
    """
    Wrapper function that launches the fused kernel.
    Handles any 3D tensor with shape [batch, height, width].
    """
    # Reshape to 2D for processing: [batch * height, width]
    original_shape = input_tensor.shape
    batch, height, width = original_shape
    
    n_rows = batch * height
    n_cols = width
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Configure grid and block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    # Launch kernel
    fused_max_sub_softmax_kernel[grid](
        input_tensor,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_max_sub_softmax