import torch
import triton
import triton.language as tl

NEG_INF = -3.4028234663852886e+38

@triton.jit
def attention_mask_kernel(
    in_ptr,
    out_ptr,
    seq_len,
    neg_inf,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total = seq_len * seq_len
    valid = offsets < total
    
    i = offsets // seq_len
    j = offsets % seq_len
    
    # Causal mask: allowed if j <= i (query position i can attend to key position j)
    causal_allowed = j <= i
    
    # Load padding mask value at position j (in_0 has shape (1, seq_len), int64 dtype)
    # Access in_0[0, j] = flat offset j (since batch_size=1 and row-major)
    padding_val = tl.load(in_ptr + j, mask=(j < seq_len) & valid, other=1)
    padding_allowed = padding_val != 0
    
    # Combined: allowed only if both causal and padding allow it
    allowed = causal_allowed & padding_allowed
    result = tl.where(allowed, 0.0, neg_inf)
    
    tl.store(out_ptr + offsets, result, mask=valid)

@torch.fx.wrap
def fused_attention_mask(in_0):
    seq_len = in_0.shape[-1]
    out = torch.empty((1, 1, seq_len, seq_len), dtype=torch.float32, device=in_0.device)
    total = seq_len * seq_len
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    attention_mask_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        seq_len=seq_len,
        neg_inf=NEG_INF,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out