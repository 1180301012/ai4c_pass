import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, device):
    tmp_4 = torch.arange(0, 9, dtype=torch.int64, device=device)
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_6 = tmp_5
    tmp_7 = tmp_6.view(-1)
    tmp_8 = in_1.index_select(0, tmp_7)
    tmp_9 = tmp_8.view(1, 9, 1024)
    return tmp_9

# Argument extraction function
def replacement_args(in_1, device):
    return (in_1,)

# Triton kernel
@triton.jit
def gather_kernel(
    in_ptr,
    out_ptr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    # Each block handles one row of the output (rows 2-10 in in_1)
    row_idx = tl.program_id(0)
    # The actual row in in_1 for this output row
    start_row = row_idx + 2
    
    # Block size for hidden dimension
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Load from in_1 (start_row, all cols), store to out_1 (row_idx, all cols)
    in_offsets = start_row * hidden_size + offsets
    in_data = tl.load(in_ptr + in_offsets, mask=mask)
    out_offsets = row_idx * hidden_size + offsets
    tl.store(out_ptr + out_offsets, in_data, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_gather(in_1):
    batch_size, hidden_size = in_1.shape
    # Output shape: [1, 9, hidden_size]
    out = torch.empty((1, 9, hidden_size), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 128
    grid = (9, (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    gather_kernel[grid](
        in_ptr=in_1,
        out_ptr=out,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Replacement function (returns the function, not the call)
def replacement_func():
    return optimized_gather