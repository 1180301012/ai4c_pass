import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_1 = x.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(x):
    return (x,)

@triton.jit
def prefix_scan_kernel(
    mask_ptr, out_ptr,
    batch_size, seq_len,
    BLOCK_SEQ: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    block_start = pid * BLOCK_SEQ
    block_end = tl.minimum(block_start + BLOCK_SEQ, seq_len)
    offsets = block_start + tl.arange(0, BLOCK_SEQ)
    mask = tl.load(mask_ptr + batch_id * seq_len + offsets, mask=offsets < seq_len, other=0)
    
    # Compute local prefix sum within block
    cumsum = tl.zeros(BLOCK_SEQ, dtype=tl.int32)
    for i in range(BLOCK_SEQ):
        if i == 0:
            cumsum[i] = mask[i]
        else:
            cumsum[i] = cumsum[i-1] + mask[i]
    
    # Apply mask: result = cumsum * mask
    out = cumsum * mask
    
    tl.store(out_ptr + batch_id * seq_len + offsets, out, mask=offsets < seq_len)

@torch.fx.wrap
def optimized_pass(x):
    mask = (x != 1).int()
    batch, seq_len = mask.shape
    out = torch.empty_like(mask, dtype=torch.int32)
    
    BLOCK_SEQ = 1024  # Matches max sequence length seen in inputs
    num_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    prefix_scan_kernel[(num_blocks, batch)](
        mask_ptr=mask,
        out_ptr=out,
        batch_size=batch,
        seq_len=seq_len,
        BLOCK_SEQ=BLOCK_SEQ,
    )
    
    return out + 1

def replacement_func():
    return optimized_pass