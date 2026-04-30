import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match pattern: softmax -> dropout(0.0)
    
    This is the attention softmax computation pattern.
    Dropout with p=0.0 is a no-op, so we just need to optimize softmax.
    
    Returns: result after dropout
    """
    tmp_21 = torch.nn.functional.softmax(x, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22


def replacement_args(x):
    return (x,)


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    num_rows: tl.constexpr,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for softmax computation.
    
    Input: flattened tensor with shape (num_rows, row_size)
    Output: softmax result (same shape)
    
    softmax is computed row-wise: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    pid = tl.program_id(0)
    
    # Each program processes one row
    row_offset = pid * row_size
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * row_size
    
    # Load the row
    row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exp(x - max)
    row_minus_max = row - row_max
    exp_row = tl.exp(row_minus_max)
    
    # Sum of exp values
    exp_sum = tl.sum(exp_row, axis=0)
    
    # Softmax result
    softmax_row = exp_row / exp_sum
    
    # Store result
    tl.store(output_ptr + offsets, softmax_row, mask=mask)


@torch.fx.wrap
def fused_softmax(x):
    """
    Fused softmax operation.
    
    Input: x with shape (N, num_heads, 64, 64) - already reshaped
    Output: softmax result over last dimension
    """
    # Determine dimensions - input is (N, num_heads, 64, 64)
    N = x.shape[0]
    num_heads = x.shape[1]
    M, row_size = x.shape[2], x.shape[3]  # 64, 64
    
    num_rows = N * num_heads  # Each (64,) row gets its own program
    
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    num_programs = num_rows
    
    fused_softmax_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        num_rows=num_rows,
        row_size=row_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_softmax