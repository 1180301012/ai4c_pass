import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    batch_seq = batch_size * seq_len
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    row_start = row * BLOCK_SIZE
    col_start = col * BLOCK_SIZE
    
    x_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE):
        k_block = min(k + BLOCK_SIZE, in_features)
        x_load = x_ptr + row_start * in_features + k + tl.arange(0, BLOCK_SIZE)[:, None]
        mask = (tl.arange(0, BLOCK_SIZE)[:, None] < in_features) & (tl.arange(0, BLOCK_SIZE) < k_block)
        x_block += tl.load(x_load, mask=mask, other=0.0)
        
    w_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE):
        k_block = min(k + BLOCK_SIZE, in_features)
        w_load = w_ptr + col_start * in_features + k + tl.arange(0, BLOCK_SIZE)[None, :]
        mask = (tl.arange(0, BLOCK_SIZE)[None, :] < in_features) & (tl.arange(0, BLOCK_SIZE) < k_block)
        w_block += tl.load(w_load, mask=mask, other=0.0)
        
    out = tl.dot(x_block, w_block)
    
    bias = tl.load(bias_ptr + col_start, mask=col_start < out_features)
    out += bias
    
    out_store = out_ptr + row_start * out_features + col_start
    tl.store(out_store, out, mask=(row_start < batch_seq) & (col_start < out_features))

@torch.fx.wrap
def linear_wrapper(in_2, in_1, in_0):
    batch_size, seq_len, in_features = in_2.shape
    out_features = in_1.shape[0]
    
    x_2d = in_2.view(-1, in_features)
    out_2d = torch.empty((batch_size * seq_len, out_features), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(batch_size * seq_len, BLOCK_SIZE), triton.cdiv(out_features, BLOCK_SIZE))
    
    linear_kernel[grid](
        x_ptr=x_2d, w_ptr=in_1, bias_ptr=in_0, out_ptr=out_2d,
        batch_size=batch_size, seq_len=seq_len, in_features=in_features, out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_2d.view(batch_size, seq_len, out_features)

def wrapped(in_2, in_1, in_0):
    linear_result = linear_wrapper(in_2, in_1, in_0)
    return linear_result, linear_result.transpose(1, 2)

def replacement_func():
    return wrapped