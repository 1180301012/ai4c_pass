import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_reduce(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    batch_size: tl.int32,
    seq_len: tl.int32,
    num_features: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    feature = tl.program_id(0)
    start_seq = tl.program_id(1) * BLOCK_SIZE
    seq = start_seq + tl.arange(0, BLOCK_SIZE)
    mask = seq < seq_len

    # Calculate offset for this feature in the input tensor
    offsets = seq * num_features + feature
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x

    sum_val = tl.sum(x, axis=0)
    sum_sq_val = tl.sum(x_sq, axis=0)

    tl.atomic_add(sum_ptr + feature, sum_val)
    tl.atomic_add(sum_sq_ptr + feature, sum_sq_val)

@triton.jit
def layer_norm_normalize(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size: tl.int32,
    seq_len: tl.int32,
    num_features: tl.int32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    feature = tl.program_id(0)
    start_seq = tl.program_id(1) * BLOCK_SIZE
    seq = start_seq + tl.arange(0, BLOCK_SIZE)
    mask = seq < seq_len

    # Load per-feature parameters
    mean_val = tl.load(mean_ptr + feature)
    var_val = tl.load(var_ptr + feature)
    weight_val = tl.load(weight_ptr + feature)
    bias_val = tl.load(bias_ptr + feature)

    denom = tl.sqrt(var_val + eps)

    offsets = seq * num_features + feature
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_norm = (x - mean_val) / denom
    x_out = x_norm * weight_val + bias_val

    tl.store(out_ptr + offsets, x_out, mask=mask)

@torch.fx.wrap
def layer_norm(x, weight, bias, eps):
    batch_size, seq_len, num_features = x.shape
    sum_ = torch.empty(num_features, dtype=x.dtype, device=x.device)
    sum_sq = torch.empty(num_features, dtype=x.dtype, device=x.device)
    sum_.zero_()
    sum_sq.zero_()

    BLOCK_SIZE = 128
    grid = (num_features, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    layer_norm_reduce[grid](
        x_ptr=x,
        sum_ptr=sum_,
        sum_sq_ptr=sum_sq,
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        BLOCK_SIZE=BLOCK_SIZE
    )

    mean = sum_ / seq_len
    var = (sum_sq / seq_len) - (mean * mean)

    out = torch.empty_like(x)
    layer_norm_normalize[grid](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def pattern(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, (384,), weight, bias, eps)

def replacement_args(x, weight, bias, eps):
    return (x, weight, bias, eps)

def replacement_func():
    return layer_norm