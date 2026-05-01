import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1):
    tmp0 = torch.max(in_0, -1, keepdim=True)
    tmp1 = tmp0[0]
    tmp2 = tmp1.expand_as(in_0)
    tmp3 = tmp2 - in_0
    tmp4 = torch.nn.functional.softmax(tmp3, dim=-1)
    return tmp4

# Argument extraction function

def replacement_args(in_0, in_1):
    return (in_0,)

# Optimized kernel
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    # Block index for (B, C)
    row_idx = tl.program_id(0)
    b = row_idx // C
    c = row_idx % C
    
    # Calculate start index for this row
    x_start = b * C * S + c * S

    # Each thread processes one element in the sequence
    k = tl.thread_id(0)
    
    # Load input element
    x = tl.load(x_ptr + x_start + k)

    # Compute max value for the row using warp reduction
    max_val = x
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        other = tl.shfl_xor(x, i)
        max_val = tl.max(max_val, other)

    # Compute exponential
    exp_x = tl.exp(x - max_val)

    # Compute sum of exponentials for the row
    sum_exp = exp_x
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        other = tl.shfl_xor(exp_x, i)
        sum_exp = sum_exp + other

    # Compute softmax
    out_val = exp_x / sum_exp
    
    # Store output
    tl.store(out_ptr + x_start + k, out_val)

# Kernel wrapper
@torch.fx.wrap
def softmax_wrapper(x):
    # Get dimensions
    B = x.shape[0]
    C = x.shape[1]
    S = x.shape[2]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Configure kernel grid and block size
    grid = (B * C,)
    BLOCK_SIZE = S
    
    # Launch kernel
    softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        B=B,
        C=C,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function

def replacement_func():
    return softmax_wrapper