import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def cumsum_with_mask_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_start = batch_idx * seq_len
    
    pos = tl.thread_id(0)
    if pos >= seq_len:
        return
    
    in_1_val = tl.load(in_1_ptr + seq_start + pos)
    in_0_val = tl.load(in_0_ptr + seq_start + pos)
    
    if pos == 0:
        current_sum = 1 if in_0_val == 0 else in_1_val
    else:
        prev_sum = tl.load(out_ptr + seq_start + pos - 1)
        current_sum = 1 if in_0_val == 0 else prev_sum + in_1_val
    
    tl.store(out_ptr + seq_start + pos, current_sum)

@torch.fx.wrap
def cumsum_with_mask(in_0, in_1):
    batch_size, seq_len = in_1.shape
    out = torch.empty_like(in_1)
    
    BLOCK_SIZE = 1024
    num_blocks = batch_size
    
    cumsum_with_mask_kernel[(num_blocks,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return cumsum_with_mask