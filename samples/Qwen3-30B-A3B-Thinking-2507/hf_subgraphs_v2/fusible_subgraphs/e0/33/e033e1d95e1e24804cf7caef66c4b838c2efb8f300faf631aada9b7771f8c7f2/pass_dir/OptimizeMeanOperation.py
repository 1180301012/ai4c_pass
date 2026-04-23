import torch
import triton
import triton.language as tl

# Pattern matching for mean operation with keepdim=True
def pattern(a):
    return a.mean(dim=-2, keepdim=True)

# Extract arguments from pattern match
def replacement_args(a):
    return (a,)

# Triton kernel for efficient mean computation
@triton.jit
def mean_kernel(
    X_ptr,
    Y_ptr,
    batch,
    n,
    m,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    base_ptr = X_ptr + batch_idx * n * m + channel_idx
    for i in range(0, n, BLOCK_SIZE):
        block_ptr = tl.make_block_ptr(
            base=base_ptr,
            shape=(n,),
            strides=(m,),
            offsets=(i,),
            block_shape=(BLOCK_SIZE,),
            order=(0,)
        )
        x = tl.load(block_ptr)
        sum_val = sum_val + tl.sum(x)
    
    mean_val = tl.sum(sum_val) / n
    tl.store(Y_ptr + batch_idx * m + channel_idx, mean_val)

# Wrapper function for the Triton kernel
@torch.fx.wrap
def mean_wrapper(X):
    batch, n, m = X.shape
    Y = torch.empty((batch, m), dtype=X.dtype, device=X.device)
    
    grid = (batch, m)
    BLOCK_SIZE = 128
    
    mean_kernel[grid](X, Y, batch, n, m, BLOCK_SIZE=BLOCK_SIZE)
    
    Y = Y.reshape(batch, 1, m)
    return Y

def replacement_func():
    return mean_wrapper