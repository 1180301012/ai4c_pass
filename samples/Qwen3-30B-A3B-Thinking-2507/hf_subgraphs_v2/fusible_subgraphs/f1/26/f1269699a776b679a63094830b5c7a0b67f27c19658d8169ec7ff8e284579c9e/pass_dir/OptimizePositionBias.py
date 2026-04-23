import torch
import triton
import triton.language as tl

def pattern(in_0):
    seq_len = in_0.shape[1]
    tmp_10 = torch.arange(seq_len, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(seq_len, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    result = tmp_19 + tmp_31
    return result

def replacement_args(in_0):
    seq_len = in_0.shape[1]
    return (seq_len,)

@triton.jit
def position_bias_kernel(
    pos_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    pos_i = tl.load(pos_ptr + i)
    pos_j = tl.load(pos_ptr + j)
    
    d = tl.abs(pos_i - pos_j)
    
    forward_bias = 16 * (i < j)
    
    d_lt_8 = d < 8
    d_float = d.to(tl.float32)
    d_over_8 = d_float / 8.0
    log_val = tl.log(d_over_8)
    transformed = 8.0 + 8.0 * (log_val / 2.772588722239781)
    transformed = tl.floor(transformed)
    transformed = tl.minimum(transformed, 15.0)
    
    value = tl.where(d_lt_8, d, transformed)
    
    result = forward_bias + value
    tl.store(out_ptr + i * N + j, result)

@torch.fx.wrap
def position_bias(in_0):
    seq_len = in_0.shape[1]
    pos = torch.arange(seq_len, dtype=torch.int64)
    out = torch.empty((seq_len, seq_len), dtype=torch.int64)
    
    grid = (seq_len, seq_len)
    
    position_bias_kernel[grid](
        pos_ptr=pos,
        out_ptr=out,
        N=seq_len,
        BLOCK_SIZE=32
    )
    
    return out

def replacement_func():
    return position_bias