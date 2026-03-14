import torch
import triton
import triton.language as tl

# Pattern function - must exactly match the computation in model.py
def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def scale_softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    pid = tl.program_id(0)
    
    # Direct row offset (assumes contiguous input)
    row_offset = pid * N
    
    # Load row and apply scale
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(input_ptr + row_offset + offs, mask=mask, other=-float('inf'))
    x = x * scale
    
    # Softmax computation
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_out = x_exp / x_sum
    
    # Store to same position
    tl.store(output_ptr + row_offset + offs, softmax_out, mask=mask)

# Wrapper function
@torch.fx.wrap
def scale_softmax_transpose(in_0):
    # Get shape info
    shape = in_0.shape
    B, H, M, N = shape[0], shape[1], shape[2], shape[3]
    total_rows = B * H * M
    
    # Allocate output with same layout
    out = torch.empty_like(in_0)
    
    scale_softmax_kernel[(total_rows,)](
        in_0,
        out,
        N,
        0.1767766952966369,
        BLOCK_SIZE=512,
        num_warps=2,
        num_stages=2,
    )
    
    # Return transposed view (no memory copy)
    return out.transpose(-2, -1)

# Replacement function - returns the callable
def replacement_func():
    return scale_softmax_transpose