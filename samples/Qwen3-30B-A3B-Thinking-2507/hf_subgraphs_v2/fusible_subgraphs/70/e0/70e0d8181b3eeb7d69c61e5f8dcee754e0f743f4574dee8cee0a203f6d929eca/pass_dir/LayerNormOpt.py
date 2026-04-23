import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    eps = 1e-05
    return torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, eps)

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0, 1e-05)

@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, seq, feat,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0)
    batch_idx = idx // seq
    seq_idx = idx % seq
    f_idx = tl.arange(0, BLOCK_SIZE)
    
    x_start = x_ptr + batch_idx * seq * feat + seq_idx * feat
    x = tl.load(x_start + f_idx, mask=f_idx < feat, other=0.0)
    
    sum_val = tl.zeros((1,), dtype=tl.float32)
    sum_squares = tl.zeros((1,), dtype=tl.float32)
    sum_val += tl.sum(x, axis=0)
    sum_squares += tl.sum(x * x, axis=0)
    
    mean = sum_val / tl.cast(feat, tl.float32)
    var = sum_squares / tl.cast(feat, tl.float32) - mean * mean
    var = tl.sqrt(tl.maximum(var, 0.0) + eps)
    
    weight = tl.load(w_ptr + f_idx, mask=f_idx < feat, other=0.0)
    bias = tl.load(b_ptr + f_idx, mask=f_idx < feat, other=0.0)
    
    x_normalized = (x - mean) / var
    out = x_normalized * weight + bias
    
    out_start = out_ptr + batch_idx * seq * feat + seq_idx * feat
    tl.store(out_start + f_idx, out, mask=f_idx < feat)

@torch.fx.wrap
def layer_norm(x, weight, bias, eps):
    batch, seq, feat = x.shape
    grid = (batch * seq,)
    BLOCK_SIZE = feat
    out = torch.empty_like(x)
    layer_norm_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        b_ptr=bias,
        out_ptr=out,
        batch=batch,
        seq=seq,
        feat=feat,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return layer_norm