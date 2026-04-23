import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (128,), weight, bias, 1e-05)

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    batch,
    seq,
    features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current block's position
    block_id = tl.program_id(0)
    batch_idx = block_id // seq
    seq_idx = block_id % seq

    # Compute the base pointer for the input and output
    x_base = x_ptr + batch_idx * x_stride_0 + seq_idx * x_stride_1
    out_base = out_ptr + batch_idx * out_stride_0 + seq_idx * out_stride_1

    # Load the entire feature vector
    x = tl.load(x_base + tl.arange(0, features), mask=tl.arange(0, features) < features)

    # Compute mean
    mean = tl.sum(x, axis=0) / features

    # Compute variance
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    var = (sum_sq - mean * mean) / features

    # Normalize
    x_norm = (x - mean) * tl.rsqrt(var + eps)

    # Apply weight and bias
    if weight_ptr != 0:
        weight = tl.load(weight_ptr + tl.arange(0, features))
        x_norm = x_norm * weight
    if bias_ptr != 0:
        bias = tl.load(bias_ptr + tl.arange(0, features))
        x_norm = x_norm + bias

    # Store the normalized vector
    tl.store(out_base + tl.arange(0, features), x_norm)

# Wrapper function
@torch.fx.wrap
def layer_norm_optimized(x, weight, bias):
    batch, seq, features = x.shape
    x_stride_0, x_stride_1, x_stride_2 = x.stride()
    out = torch.empty_like(x)
    out_stride_0, out_stride_1, out_stride_2 = out.stride()

    num_blocks = batch * seq
    BLOCK_SIZE = features  # One block per feature vector

    layer_norm_kernel[(num_blocks,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        x_stride_0=x_stride_0,
        x_stride_1=x_stride_1,
        x_stride_2=x_stride_2,
        out_stride_0=out_stride_0,
        out_stride_1=out_stride_1,
        out_stride_2=out_stride_2,
        batch=batch,
        seq=seq,
        features=features,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Replacement function
def replacement_func():
    return layer_norm_optimized