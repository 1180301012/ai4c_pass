import torch
import triton
import triton.language as tl

def pattern(x, y):
    s = torch.nn.functional.softmax(x, dim=-1)
    m = torch.matmul(s, y)
    p = m.permute(0, 2, 1, 3)
    return p

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_softmax_matmul_kernel(
    x_ptr, y_ptr, out_ptr,
    batch, heads, seq_len, head_dim,
    BLOCK_SEQ: tl.constexpr = 32,
    BLOCK_HEAD_DIM: tl.constexpr = 32,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    x_offset = batch_idx * heads * seq_len * seq_len + head_idx * seq_len * seq_len + seq_idx * seq_len
    x_ptr_batch_head_seq = x_ptr + x_offset
    
    y_offset = batch_idx * heads * seq_len * head_dim + head_idx * seq_len * head_dim
    y_ptr_batch_head = y_ptr + y_offset
    
    out_offset = batch_idx * seq_len * heads * head_dim + seq_idx * heads * head_dim + head_idx * head_dim
    out_ptr_batch_seq_head = out_ptr + out_offset
    
    x = tl.load(x_ptr_batch_head_seq + tl.arange(0, seq_len), mask=tl.arange(0, seq_len) < seq_len)
    x_exp = tl.exp(x)
    x_exp_sum = tl.sum(x_exp, axis=0)
    
    y = tl.load(y_ptr_batch_head + tl.arange(0, seq_len * head_dim), mask=tl.arange(0, seq_len * head_dim) < seq_len * head_dim)
    y = y.reshape(seq_len, head_dim)
    
    numerator = x_exp[:, None] * y
    numerator_sum = tl.sum(numerator, axis=0)
    result = numerator_sum / x_exp_sum
    
    tl.store(out_ptr_batch_seq_head + tl.arange(0, head_dim), result, mask=tl.arange(0, head_dim) < head_dim)

@torch.fx.wrap
def fused_softmax_matmul(x, y):
    batch, heads, seq_len, _ = x.shape
    _, _, _, head_dim = y.shape
    
    out = torch.empty((batch, seq_len, heads, head_dim), dtype=x.dtype, device=x.device)
    
    BLOCK_SEQ = 32
    BLOCK_HEAD_DIM = 32
    
    grid = (batch, heads, seq_len)
    
    fused_softmax_matmul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
    )
    
    return out

def replacement_func():
    return fused_softmax_matmul