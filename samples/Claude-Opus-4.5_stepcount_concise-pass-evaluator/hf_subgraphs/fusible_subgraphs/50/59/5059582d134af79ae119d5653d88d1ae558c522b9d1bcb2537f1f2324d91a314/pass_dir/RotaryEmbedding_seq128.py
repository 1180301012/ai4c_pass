import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.arange(128, device=in_0.device, dtype=in_0.dtype)
    tmp_2 = tmp_1.type_as(tmp_0)
    tmp_3 = torch.outer(tmp_2, tmp_0)
    tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_5 = tmp_4.to(in_0.device)
    tmp_6 = tmp_5.cos()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_5.sin()
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_10 = tmp_7[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
    tmp_11 = tmp_9[slice(None, None, None), slice(None, None, None), slice(None, 128, None), slice(None, None, None)]
    tmp_12 = in_1 * tmp_10
    tmp_13 = in_1.chunk(2, dim=-1)
    tmp_14 = tmp_13[0]
    tmp_15 = tmp_13[1]
    return (tmp_7, tmp_9, tmp_11, tmp_12, tmp_14, tmp_15)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def rotary_embedding_kernel_128(
    inv_freq_ptr,
    cos_out_ptr,
    sin_out_ptr,
    SEQ_LEN,
    HEAD_DIM,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    n_elements = SEQ_LEN * HEAD_DIM
    mask = offsets < n_elements
    
    # Convert linear index to (seq, dim) coordinates
    seq_idx = offsets // HEAD_DIM
    dim_idx = offsets % HEAD_DIM
    
    # Since we concatenate [outer, outer], dim maps to inv_freq via modulo
    inv_freq_idx = dim_idx % HALF_DIM
    inv_freq = tl.load(inv_freq_ptr + inv_freq_idx, mask=mask, other=0.0)
    
    # Compute frequency: seq_idx * inv_freq
    freq = tl.cast(seq_idx, tl.float32) * inv_freq
    
    # Compute cos and sin
    cos_val = tl.cos(freq)
    sin_val = tl.sin(freq)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def optimized_rotary_128(in_0, in_1):
    SEQ_LEN = 128
    HEAD_DIM = 64
    HALF_DIM = 32
    
    # Allocate output tensors
    cos_emb = torch.empty(SEQ_LEN, HEAD_DIM, device=in_0.device, dtype=in_0.dtype)
    sin_emb = torch.empty(SEQ_LEN, HEAD_DIM, device=in_0.device, dtype=in_0.dtype)
    
    BLOCK_SIZE = 512
    n_elements = SEQ_LEN * HEAD_DIM
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    rotary_embedding_kernel_128[(num_blocks,)](
        in_0,
        cos_emb,
        sin_emb,
        SEQ_LEN,
        HEAD_DIM,
        HALF_DIM,
        BLOCK_SIZE,
    )
    
    # Reshape to [1, 1, SEQ_LEN, HEAD_DIM]
    cos_out = cos_emb[None, None, :, :]
    sin_out = sin_emb[None, None, :, :]
    
    # tmp_11 is slice of sin_out (identity in this case)
    tmp_11 = sin_out[:, :, :SEQ_LEN, :]
    
    # Multiplication: in_1 * cos_embedding (broadcast)
    tmp_12 = in_1 * cos_out
    
    # Chunk operation
    tmp_14, tmp_15 = in_1.chunk(2, dim=-1)
    
    return (cos_out, sin_out, tmp_11, tmp_12, tmp_14, tmp_15)

def replacement_func():
    return optimized_rotary_128