import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    seq_len, hidden_size, eps,
    BLOCK_SIZE: tl.constexpr
):
    seq_id = tl.program_id(0)
    block_start = seq_id * hidden_size
    hidden_id = tl.thread_id(0)
    x_block = x_ptr + block_start
    out_block = out_ptr + block_start
    
    sum = 0.0
    sum_sq = 0.0
    for i in range(hidden_size):
        x_val = tl.load(x_block + i)
        sum += x_val
        sum_sq += x_val * x_val
    
    mean = sum / hidden_size
    var = sum_sq / hidden_size - mean * mean
    inv_sqrt_var = 1.0 / tl.sqrt(var + eps)
    
    x_val = tl.load(x_block + hidden_id)
    y = (x_val - mean) * inv_sqrt_var * tl.load(weight_ptr + hidden_id) + tl.load(bias_ptr + hidden_id)
    tl.store(out_block + hidden_id, y)

def pattern(tmp_7, in_1, in_2):
    return torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)

def replacement_args(tmp_7, in_1, in_2):
    return (tmp_7, in_1, in_2)

@torch.fx.wrap
def layer_norm_triton(x, weight, bias):
    batch, seq_len, hidden_size = x.shape
    eps = 1e-05
    out = torch.empty_like(x)
    
    grid = (seq_len,)
    BLOCK_SIZE = hidden_size
    
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return layer_norm_triton