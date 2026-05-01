import torch
import triton
import triton.language as tl

# LayerNorm pattern matching
# Exact match: torch.nn.functional.layer_norm(input, (512,), weight, bias, 1e-05)
def pattern(in_2, in_1, in_0):
    return torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)

def replacement_args(in_2, in_1, in_0):
    # Extract all required arguments
    return (in_2, in_1, in_0)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    n_seq,
    BLOCK_SIZE: tl.constexpr,
):
    # One block per channel (512 channels)
    channel = tl.program_id(0)
    start_idx = channel * n_seq

    # Sum and sum_sq for current channel (for mean/var computation)
    sum_val = 0.0
    sum_sq = 0.0

    # Compute mean/var: sum over all elements in the channel's sequence
    for i in range(0, n_seq, BLOCK_SIZE):
        idx = i + tl.thread_id(0)
        if idx < n_seq:
            x = tl.load(x_ptr + start_idx + idx)
            sum_val += x
            sum_sq += x * x

    # Reduce within block
    sum_val = tl.sum(sum_val, axis=0)
    sum_sq = tl.sum(sum_sq, axis=0)

    # Compute mean, variance, and inverse standard deviation
    mean = sum_val / n_seq
    var = (sum_sq / n_seq) - (mean * mean)
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Apply layer norm: (x - mean) * inv_std * weight + bias
    for i in range(0, n_seq, BLOCK_SIZE):
        idx = i + tl.thread_id(0)
        if idx < n_seq:
            x = tl.load(x_ptr + start_idx + idx)
            normalized = (x - mean) * inv_std
            y = normalized * tl.load(weight_ptr + channel) + tl.load(bias_ptr + channel)
            tl.store(out_ptr + start_idx + idx, y)

@torch.fx.wrap
def layer_norm_wrapper(in_2, in_1, in_0):
    # Fixed dimensions from weight_meta.py
    n_channels = 512
    n_seq = 3999
    BLOCK_SIZE = 256

    # Output shape: [1, 3999, 512] same as input
    out = torch.empty_like(in_2)

    # Launch kernel: 512 blocks (one per channel), each with 256 threads
    layer_norm_kernel[(n_channels,)](
        x_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        n_channels=n_channels,
        n_seq=n_seq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return layer_norm_wrapper