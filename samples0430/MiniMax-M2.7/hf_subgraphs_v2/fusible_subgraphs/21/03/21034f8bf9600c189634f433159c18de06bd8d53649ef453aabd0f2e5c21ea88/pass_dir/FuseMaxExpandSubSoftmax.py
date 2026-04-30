import torch
import triton
import triton.language as tl

# Autotune configuration for optimal performance
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
    ],
    key=["cols"],
)
@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    rows_strides,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: max -> expand -> subtract -> softmax
    
    This fuses the numerical stable softmax computation into a single kernel.
    """
    # Get row information (each program handles one row)
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_offset = row_idx * rows_strides
    
    # Create offsets for all columns in this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = row_offset + col_offsets
    mask = col_offsets < cols
    
    # Load values
    vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
    
    # Compute max for numerical stability
    max_val = tl.max(vals)
    
    # Compute exp(max - x) for all elements
    exp_vals = tl.exp(vals - max_val)
    
    # Sum of exponentials for normalization
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax output
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_softmax_wrapper(in_0, in_1):
    """
    Fused implementation of:
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(...)
    
    Returns: (tmp_4, tmp_5)
    """
    batch_size, seq_len, _ = in_0.shape
    
    # Allocate output for softmax
    out_softmax = torch.empty_like(in_0)
    
    # Launch Triton kernel - one program per row
    grid = (batch_size * seq_len,)
    
    fused_softmax_kernel[grid](
        in_0,
        out_softmax,
        in_0.stride(0),  # rows_strides
        in_0.shape[-1],  # cols
    )
    
    # Compute view output - infer dimensions from tensor shapes
    # in_0 shape is [B, 512, 512], first dim of in_1 is same B
    view_dim0 = in_1.shape[0]
    # Second view dim is always 512
    view_dim1 = 512
    out_view = in_1.view(view_dim0, view_dim1, -1)
    
    return (out_softmax, out_view)


def pattern(in_0, in_1):
    """
    Match the pattern:
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(dim0, 512, -1)
    
    Returns: (tmp_4, tmp_5)
    """
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(in_1.shape[0], 512, -1)
    return (tmp_4, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_wrapper