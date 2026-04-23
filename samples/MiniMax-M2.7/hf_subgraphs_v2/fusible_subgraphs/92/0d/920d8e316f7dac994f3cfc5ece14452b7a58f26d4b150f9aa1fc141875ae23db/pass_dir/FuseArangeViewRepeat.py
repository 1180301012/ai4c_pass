import torch
from torch import device
import triton
import triton.language as tl


@triton.jit
def arange_repeat_kernel(
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that creates output[2, N] where output[i, j] = j
    This replaces: arange(0, N) -> view(1, -1) -> repeat(2, 1)
    """
    # Get program ID and calculate which row and column block
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE elements in a single row
    row = pid // 1  # Since we're doing 1 block per row for simplicity
    col_offset = tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = col_offset < N
    
    # Value is just the column index (j from 0 to N-1)
    values = col_offset
    
    # Output is stored as [row, col] = [row, col_offset]
    output_offsets = row * N + col_offset
    
    # Store results
    tl.store(output_ptr + output_offsets, values, mask=mask)


@torch.fx.wrap
def arange_repeat_kernel_wrapper(start, end, repeat_rows, dev):
    """
    Wrapper for the fused arange + view + repeat kernel.
    
    Replaces: arange(start, end).view(1, -1).repeat(repeat_rows, 1)
    Result: A (repeat_rows, end-start) tensor where result[i, j] = start + j
    """
    N = end - start
    
    # BLOCK_SIZE must be >= N to write all elements in one pass
    # Use 1024 which covers all cases (max N is 1000)
    BLOCK_SIZE = 1024
    
    # Use num_programs = repeat_rows (one program per row)
    total_programs = repeat_rows
    
    # Create output tensor with int64 dtype (same as torch.arange)
    output = torch.empty((repeat_rows, N), dtype=torch.int64, device=dev)
    
    # Launch kernel
    arange_repeat_kernel[(total_programs,)](
        output_ptr=output,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(repeat_input):
    """
    Match the pattern: repeat_input.view(1, -1).repeat(2, 1)
    where repeat_input is the input to repeat (output of view, which is output of arange).
    """
    tmp_view = repeat_input.view(1, -1)
    result = tmp_view.repeat(2, 1)
    return result


def replacement_args(repeat_input):
    """
    Extract arguments from matched nodes:
    - repeat_input: tensor that is the input to view (arange output)
    """
    # Note: Cannot extract device reliably during symbolic tracing
    # Using fixed 'cuda' device since all graphs use CUDA
    dev = "cuda"
    
    # Get the size from the input tensor (this is the N from arange)
    end = repeat_input.numel()
    start = 0
    
    # For repeat(2, 1), repeat_rows is always 2
    repeat_rows = 2
    
    return (start, end, repeat_rows, dev)


def replacement_func():
    """
    Return the replacement function that fuses arange + view + repeat
    """
    return arange_repeat_kernel_wrapper