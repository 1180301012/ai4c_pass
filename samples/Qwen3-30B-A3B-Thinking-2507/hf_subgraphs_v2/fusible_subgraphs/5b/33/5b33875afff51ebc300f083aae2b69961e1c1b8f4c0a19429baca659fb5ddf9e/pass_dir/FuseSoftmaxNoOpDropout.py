import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: softmax(input, dim=-1) followed by dropout(p=0.1, training=False, inplace=False)
def pattern(tmp_1):
    softmax = torch.nn.functional.softmax(tmp_1, dim=-1)
    dropout = torch.nn.functional.dropout(softmax, 0.1, False, False)
    return dropout

# Argument extraction function
# Extracts the input tensor to the softmax

def replacement_args(tmp_1):
    return (tmp_1,)

# Triton kernel for softmax (with no-op dropout handled by skipping dropout)
@triton.jit
def softmax_kernel(
    x_ptr, x_stride0, x_stride1, x_stride2, x_stride3,
    out_ptr, out_stride0, out_stride1, out_stride2, out_stride3,
    N: tl.constexpr,
):
    # Compute grid index (B, H, i) for the current row
    block_idx = tl.program_id(0)
    B = block_idx // (N * N)
    H = (block_idx // N) % N
    i = block_idx % N

    # Load row of length N (softmax along last dimension)
    x = tl.load(
        x_ptr + B * x_stride0 + H * x_stride1 + i * x_stride2,
        mask=tl.arange(0, N) < N,
        other=0.0
    )

    # Compute max along row
    max_val = tl.max(x)

    # Compute exp(x - max_val)
    x_sub = x - max_val
    x_exp = tl.exp(x_sub)

    # Compute sum of exp
    x_exp_sum = tl.sum(x_exp)

    # Compute softmax
    softmax = x_exp / x_exp_sum

    # Store result
    tl.store(
        out_ptr + B * out_stride0 + H * out_stride1 + i * out_stride2,
        softmax,
        mask=tl.arange(0, N) < N
    )

# Triton kernel wrapper
@torch.fx.wrap
def softmax_wrapper(x):
    # Extract dimensions (B, H, N1, N2)
    B, H, N1, N2 = x.shape
    N = N2
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel: one block per (B, H, i) row
    num_blocks = B * H * N1
    softmax_kernel[(num_blocks,)](
        x, x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N
    )
    return out

# Replacement function

def replacement_func():
    return softmax_wrapper