import torch
import triton
import triton.language as tl



# Pattern matching function
def pattern(x, y):
    return torch.matmul(x, y)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_matmul_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    # Each program handles one row of the output [2, 1]
    pid = tl.program_id(0)
    m = pid
    
    # Accumulator for this row
    acc = 0.0
    
    # Vectorized multiplication-add for the entire K dimension
    # x_ptr[m, :] * y_ptr[:, 0] - sum over k
    x = tl.load(x_ptr + m * K + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)
    y = tl.load(y_ptr + tl.arange(0, K) * N, mask=tl.arange(0, K) < K, other=0.0)
    
    acc = tl.sum(x * y)
    
    # Store the result
    tl.store(out_ptr + m, acc)

@torch.fx.wrap
def triton_matmul(x, y):
    M, K = x.shape
    K2, N = y.shape
    
    if K != K2 or N != 1:
        raise ValueError("Matrix dimensions don't match expected [M, K] @ [K, 1] -> [M, 1]")
    
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Each program handles one row
    grid = (M,)
    
    simple_matmul_kernel[grid](
        x, y, out, M, K, N
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_matmul