import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return (tmp_8,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def masked_cumsum_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    seq_idx = (pid % ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE
    
    if batch_idx >= batch_size:
        return
    
    offsets = seq_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    input_data = tl.load(input_ptr + batch_idx * seq_len + offsets, mask=mask, other=1)
    
    is_padding = (input_data == 1)
    mask_int = tl.where(is_padding, 0, 1)
    
    cumsum = tl.cumsum(mask_int, 0)
    
    positions = tl.where(is_padding, 0, cumsum)
    positions = positions + 1
    
    tl.store(output_ptr + batch_idx * seq_len + offsets, positions, mask=mask)

@torch.fx.wrap
def triton_masked_cumsum(input_tensor):
    batch_size, seq_len = input_tensor.shape
    BLOCK_SIZE = 1024
    
    num_programs = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(input_tensor, dtype=torch.int64)
    
    masked_cumsum_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return triton_masked_cumsum