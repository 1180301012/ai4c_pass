import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-12)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // seq_len
    j = pid % seq_len

    sum_x = 0.0
    sum_x_sq = 0.0
    for k in range(hidden_size):
        idx = i * seq_len * hidden_size + j * hidden_size + k
        x_val = tl.load(x_ptr + idx, dtype=tl.float32)
        sum_x += x_val
        sum_x_sq += x_val * x_val

    mean = sum_x / hidden_size
    var = sum_x_sq / hidden_size - mean * mean

    for k in range(hidden_size):
        idx = i * seq_len * hidden_size + j * hidden_size + k
        x_val = tl.load(x_ptr + idx, dtype=tl.float32)
        normalized = (x_val - mean) / tl.sqrt(var + eps)
        w = tl.load(weight_ptr + k, dtype=tl.float32)
        b = tl.load(bias_ptr + k, dtype=tl.float32)
        out_val = normalized * w + b
        tl.store(out_ptr + idx, out_val, dtype=tl.float32)

@torch.fx.wrap
def layernorm_optimized(x, weight, bias, eps=1e-12):
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    bias_fp32 = bias.float()
    batch_size, seq_len, hidden_size = x.shape
    out_fp32 = torch.empty(batch_size, seq_len, hidden_size, dtype=torch.float32)
    num_blocks = batch_size * seq_len
    layernorm_kernel[(num_blocks,)](
        x_ptr=x_fp32,
        weight_ptr=weight_fp32,
        bias_ptr=bias_fp32,
        out_ptr=out_fp32,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=128
    )
    return out_fp32.to(x.dtype)

def replacement_func():
    return layernorm_optimized