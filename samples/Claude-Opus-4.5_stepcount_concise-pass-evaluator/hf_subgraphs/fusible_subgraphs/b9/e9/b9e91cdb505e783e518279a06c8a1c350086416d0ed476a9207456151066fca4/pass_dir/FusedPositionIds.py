import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Match the position ID computation pattern:
    1. ne(1) - create mask where input != 1
    2. int() - convert to int
    3. cumsum along dim 1
    4. type_as - convert type (no-op since already int)
    5. + 0 - add 0 (no-op)
    6. multiply by mask
    7. long() - convert to long
    8. + 1 - add 1
    """
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_position_ids_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for computing position IDs with masking.
    Each program handles one row of the input.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_len
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load input values (int64)
    input_vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=0)
    
    # Compute mask: ne(1).int() -> 1 where input != 1, 0 otherwise
    # Use int32 for faster intermediate computation
    ne_mask = input_vals != 1
    mask_vals = ne_mask.to(tl.int32)
    
    # Compute cumsum along the row
    cumsum_vals = tl.cumsum(mask_vals, axis=0)
    
    # Multiply cumsum by mask and add 1
    result = cumsum_vals * mask_vals + 1
    
    # Store result as int64
    tl.store(output_ptr + row_start + offsets, result.to(tl.int64), mask=mask)

@torch.fx.wrap
def fused_position_ids(in_0):
    """
    Wrapper function to launch the Triton kernel.
    """
    batch_size, seq_len = in_0.shape
    output = torch.empty(batch_size, seq_len, dtype=torch.int64, device=in_0.device)
    
    # Choose BLOCK_SIZE and num_warps based on seq_len
    if seq_len <= 64:
        BLOCK_SIZE = 64
        num_warps = 2
    elif seq_len <= 128:
        BLOCK_SIZE = 128
        num_warps = 4
    elif seq_len <= 256:
        BLOCK_SIZE = 256
        num_warps = 4
    elif seq_len <= 512:
        BLOCK_SIZE = 512
        num_warps = 4
    else:
        BLOCK_SIZE = 1024
        num_warps = 8
    
    # Launch one program per row
    grid = (batch_size,)
    fused_position_ids_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output

def replacement_func():
    return fused_position_ids