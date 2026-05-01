import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches the exact sequence of operations in model.py
# Note: Exclude cleanup statements like 'tmp_x = None'
def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

# Argument extraction function
# Extract the necessary inputs for the optimized kernel

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized Triton kernel for LayerNorm
@triton.jit
def layernorm_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    B,
    S,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate grid dimensions
    block_id = tl.program_id(0)
    batch_idx = block_id // S
    seq_idx = block_id % S
    
    # Calculate start index for (batch_idx, seq_idx) in X
    start = (batch_idx * S + seq_idx) * H
    
    # Shared memory for reduction
    sh_mem = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(0, H, BLOCK_SIZE):
        idx = i + tl.thread_id(0)
        if idx >= H:
            break
        x = tl.load(X_ptr + start + idx)
        sh_mem[tl.thread_id(0)] = x
        sum_val += x
        sum_sq += x * x
    sum_val = tl.sum(sum_val)
    sum_sq = tl.sum(sum_sq)
    
    # Calculate mean, variance, and std
    mean = sum_val / H
    var = sum_sq / H - mean * mean
    std = tl.sqrt(var + eps)
    
    # Compute normalized values
    for i in range(0, H, BLOCK_SIZE):
        idx = i + tl.thread_id(0)
        if idx >= H:
            break
        x = tl.load(X_ptr + start + idx)
        centered = x - mean
        normalized = centered / std
        w = tl.load(W_ptr + idx)
        b = tl.load(B_ptr + idx)
        y = normalized * w + b
        tl.store(Y_ptr + start + idx, y)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def layernorm_optimized(in_0, in_1, in_2, in_3):
    # Compute input tensor (in_3 + in_2)
    X = in_3 + in_2
    
    # Get tensor dimensions
    B, S, H = X.shape
    
    # Create output tensor
    Y = torch.empty_like(X)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks (one per (B, S) position)
    num_blocks = B * S
    
    # Launch kernel
    layernorm_kernel[(num_blocks,)](
        X_ptr=X,
        W_ptr=in_1,
        B_ptr=in_0,
        Y_ptr=Y,
        B=B,
        S=S,
        H=H,
        eps=1e-7,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return Y

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return layernorm_optimized