import torch
import triton
import triton.language as tl

def pattern(x, gamma, beta):
    return torch.nn.functional.layer_norm(x, (1024,), gamma, beta, 1e-05)

def replacement_args(x, gamma, beta):
    return (x, gamma, beta)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    batch,
    seq,
    hidden,
    eps,
):
    block_id = tl.program_id(0)
    i = block_id // seq
    j = block_id % seq
    offsets = tl.arange(0, 1024)
    offset_base = i * seq * 1024 + j * 1024
    x_data = tl.load(x_ptr + offset_base + offsets, mask=offsets < 1024)
    
    sum_x = tl.sum(x_data)
    sum_sq = tl.sum(x_data * x_data)
    
    mean = sum_x / 1024.0
    variance = (sum_sq - (sum_x * sum_x) / 1024.0) / 1024.0
    
    x_normalized = (x_data - mean) / tl.sqrt(variance + eps)
    
    gamma_data = tl.load(gamma_ptr + offsets, mask=offsets < 1024)
    beta_data = tl.load(beta_ptr + offsets, mask=offsets < 1024)
    out = gamma_data * x_normalized + beta_data
    
    tl.store(out_ptr + offset_base + offsets, out, mask=offsets < 1024)

@torch.fx.wrap
def layer_norm_custom(x, gamma, beta):
    batch = x.shape[0]
    seq = x.shape[1]
    hidden = 1024
    N = batch * seq
    out = torch.empty_like(x)
    layer_norm_kernel[(N,)](
        x_ptr=x,
        gamma_ptr=gamma,
        beta_ptr=beta,
        out_ptr=out,
        batch=batch,
        seq=seq,
        hidden=hidden,
        eps=1e-05
    )
    return out

def replacement_func():
    return layer_norm_custom