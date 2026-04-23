import torch
import triton
import triton.language as tl


# Autotune configurations for mean kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=8),
    ],
    key=['features'],
)
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean over dimension -2:
    input: [batch, seq_len, features] -> output: [batch, features]
    Each program processes one batch element.
    """
    pid_batch = tl.program_id(0)
    
    offs_feat = tl.arange(0, BLOCK_SIZE)
    mask_feat = offs_feat < features
    
    # Accumulators for each feature
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over seq_len dimension
    for k in range(seq_len):
        # Load input[b, k, :] - contiguous feature vector
        # Layout: b*seq_len*features + k*features + j
        base_offset = pid_batch * seq_len * features + k * features
        input_offsets = base_offset + offs_feat
        x = tl.load(input_ptr + input_offsets, mask=mask_feat, other=0.0)
        acc += x
    
    # Divide by seq_len
    result = acc / seq_len
    
    # Store output[b, :]
    output_offsets = pid_batch * features + offs_feat
    tl.store(output_ptr + output_offsets, result, mask=mask_feat)


def pattern(x):
    """
    Match the mean(-2) operation pattern.
    """
    return x.mean(-2)


def replacement_args(x):
    """
    Extract the tensor to compute mean on.
    """
    return (x,)


@torch.fx.wrap
def mean_wrapper(x):
    """
    Optimized mean computation using Triton.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    features = x.shape[2]
    
    output = torch.empty((batch_size, features), dtype=x.dtype, device=x.device)
    
    grid = (batch_size,)
    
    mean_kernel[grid](
        x, output,
        batch_size, seq_len, features,
    )
    
    return output


def replacement_func():
    """
    Return the optimized mean function.
    """
    return mean_wrapper