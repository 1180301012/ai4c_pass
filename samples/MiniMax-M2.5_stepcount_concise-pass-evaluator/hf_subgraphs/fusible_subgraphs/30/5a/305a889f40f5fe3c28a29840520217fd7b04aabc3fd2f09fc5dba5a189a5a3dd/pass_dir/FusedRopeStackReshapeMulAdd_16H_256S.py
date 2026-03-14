import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation from model.py
# For 16 heads, 256 seq_len
def pattern(in_1, in_3, in_5, in_6):
    """
    Pattern: -in_1 + (stack([in_1[::2], in_3[::2]], -1).reshape() * sin_emb) + mul
    """
    tmp_1 = -in_1
    tmp_2 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape((1, 16, 256, 64))
    tmp_5 = tmp_4 * in_6
    tmp_6 = in_5 + tmp_5
    return tmp_6


def replacement_args(in_1, in_3, in_5, in_6):
    return (in_1, in_3, in_5, in_6)


@triton.jit
def fused_rope_kernel(
    in_1_ptr, in_3_ptr, in_5_ptr, in_6_ptr,
    out_ptr,
    B, H, seq_len, dim,
):
    batch_idx = tl.program_id(0)
    b = batch_idx // H
    h = batch_idx % H
    
    BLOCK_SIZE: tl.constexpr = 64
    
    for start_n in range(0, seq_len, BLOCK_SIZE):
        off_n = start_n + tl.arange(0, BLOCK_SIZE)
        mask_n = off_n < seq_len
        
        for start_d in range(0, dim, BLOCK_SIZE):
            off_d = start_d + tl.arange(0, BLOCK_SIZE)
            mask_d = off_d < dim
            mask = mask_n & mask_d
            
            in_5_offset = b * H * seq_len * dim + h * seq_len * dim + off_n * dim + off_d
            in_5 = tl.load(in_5_ptr + in_5_offset, mask=mask, other=0.0)
            
            sin_emb_offset = off_n * dim + off_d
            sin_emb = tl.load(in_6_ptr + sin_emb_offset, mask=mask, other=0.0)
            
            k = off_d // 2
            is_odd = (off_d % 2) == 1
            
            in_1_offset = b * H * seq_len * (dim // 2) + h * seq_len * (dim // 2) + off_n * (dim // 2) + k
            in_1_vals = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
            
            in_3_offset = b * H * seq_len * dim + h * seq_len * dim + off_n * dim + k
            in_3_vals = tl.load(in_3_ptr + in_3_offset, mask=mask, other=0.0)
            
            rope_vals = tl.where(is_odd, in_3_vals, -in_1_vals)
            result = in_5 + rope_vals * sin_emb
            
            out_offset = b * H * seq_len * dim + h * seq_len * dim + off_n * dim + off_d
            tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_rope_wrapper(x, y, z, w):
    B = x.shape[0]
    H = x.shape[1]
    seq_len = x.shape[2]
    dim_full = y.shape[3]
    
    output = torch.empty_like(z)
    grid = (B * H,)
    
    fused_rope_kernel[grid](
        x, y, z, w,
        output,
        B, H, seq_len, dim_full,
    )
    
    return output


def replacement_func():
    return fused_rope_wrapper