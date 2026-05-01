import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, seq_len, hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    seq_id = tl.program_id(0)
    x_start = seq_id * hidden_size
    out_start = seq_id * hidden_size
    
    # Initialize sum and sum of squares to zero
    sum_val = tl.zeros((1,), dtype=tl.float32)
    sum_sq_val = tl.zeros((1,), dtype=tl.float32)
    
    # First pass: compute sum and sum of squares over hidden_size
    for off in range(0, hidden_size, BLOCK_SIZE):
        rel_off = off + tl.arange(0, BLOCK_SIZE)
        mask = rel_off < hidden_size
        x = tl.load(x_ptr + x_start + rel_off, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    
    # Reduce the sums across threads (from the current warp)
    total_sum = tl.sum(sum_val, axis=0)
    total_sum_sq = tl.sum(sum_sq_val, axis=0)
    
    # Compute mean and variance
    mean = total_sum / hidden_size
    var = (total_sum_sq / hidden_size) - (mean * mean)
    denom = tl.sqrt(var + eps)
    
    # Second pass: normalize and apply weight/bias
    for off in range(0, hidden_size, BLOCK_SIZE):
        rel_off = off + tl.arange(0, BLOCK_SIZE)
        mask = rel_off < hidden_size
        x = tl.load(x_ptr + x_start + rel_off, mask=mask, other=0.0)
        normalized = (x - mean) / denom
        weight_val = tl.load(weight_ptr + rel_off, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + rel_off, mask=mask, other=0.0)
        out = normalized * weight_val + bias_val
        tl.store(out_ptr + out_start + rel_off, out, mask=mask)

@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps=1e-05):
    # Extract seq_len and hidden_size from x (shape [1, seq_len, hidden_size])
    seq_len = x.size(1)
    hidden_size = x.size(2)
    
    # Set block size for Triton
    BLOCK_SIZE = 256
    num_blocks = seq_len
    out = torch.empty_like(x)
    
    layer_norm_kernel[(num_blocks,)](
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

def pattern(x, weight, bias):
    out = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

def replacement_func():
    return layer_norm_wrapper