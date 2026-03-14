import torch
import triton
import triton.language as tl

# Pattern matching function for the normalization sequence
def pattern(x):
    # Match the normalization operations: pow(2) + mean + add_epsilon + rsqrt
    # x corresponds to tmp_1 (in_3 + in_2)
    tmp_4 = x.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    return tmp_8, x  # return normalization factor and original input for multiplication pass

# Argument extraction function
def replacement_args(x):
    return (x,)

# Define the optimized normalization kernel
@triton.jit
def fused_norm_kernel(
    x_ptr,  # Input tensor
    out_ptr,  # Normalization factor (1/sqrt(var + epsilon))
    norm_ptr,  # Output with float dtype
    n_batch,  # Batch size
    n_seq,    # Sequence length
    n_features,  # Feature dimension (1024)
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // n_seq
    seq_id = pid % n_seq
    feature_start = tl.arange(0, REDUCE_BLOCK)
    
    # Load and compute sum of squares
    sum_sq = 0.0
    offset = batch_id * n_seq * n_features + seq_id * n_features
    for i in range(0, n_features, REDUCE_BLOCK):
        indices = feature_start + i
        mask = indices < n_features
        x = tl.load(x_ptr + offset + indices, mask=mask, other=0.0)
        sum_sq += x * x
    
    # Reduce sum_sq within the block
    if REDUCE_BLOCK > 128:
        sum_sq += tl.sum(sum_sq, axis=0)
    if REDUCE_BLOCK > 256:
        sum_sq += tl.sum(sum_sq, axis=0)
    sum_sq = tl.sum(sum_sq)
    
    # Compute variance (mean of squares)
    mean_sq = sum_sq / n_features
    
    # Compute normalization factor: 1 / sqrt(mean_sq + epsilon)
    norm_factor = tl.math.rsqrt(mean_sq + 1e-06)
    
    # Store normalization factor
    tl.store(out_ptr + batch_id * n_seq + seq_id, norm_factor)
    
    # Store the normalized input (just load and store to handle dtype conversion)
    tl.store(norm_ptr + offset + feature_start, x, mask=feature_start < n_features)

# Kernel wrapper
@torch.fx.wrap
def fused_normalization(x):
    n_batch, n_seq, n_features = x.shape
    N = n_batch * n_seq
    
    # Output for normalization factors (one per sequence element)
    norm_factors = torch.empty((n_batch, n_seq), dtype=torch.float32, device=x.device)
    
    # Output for float32 conversion (for compatibility with original)
    norm_x = torch.empty_like(x, dtype=torch.float32)
    
    # Choose block sizes for optimal performance
    BLOCK_SIZE = 128
    REDUCE_BLOCK = 256
    
    # Call the fused kernel
    fused_norm_kernel[(N,)](
        x_ptr=x,
        out_ptr=norm_factors,
        norm_ptr=norm_x,
        n_batch=n_batch,
        n_seq=n_seq,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCE_BLOCK=REDUCE_BLOCK,
    )
    
    return norm_factors, norm_x

# Replacement function
def replacement_func():
    return fused_normalization