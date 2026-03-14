import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Different tile sizes for different input sizes
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,  # position_bias
    in_1_ptr,  # scores (modified in-place)
    out_ptr,   # output
    N,         # last dimension size (softmax dim)
    M,         # product of other dims
    stride_in_0, stride_in_1, stride_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for:
    1. in_1 += in_0 (element-wise add)
    2. softmax(in_1, dim=-1)
    
    This fuses the add and softmax into a single kernel for better performance.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate row offset
    # Each program processes one row (M dimension)
    row_offset = pid
    
    # Load the row pointers
    in_0_row_ptr = in_0_ptr + row_offset * stride_in_0
    in_1_row_ptr = in_1_ptr + row_offset * stride_in_1
    out_row_ptr = out_ptr + row_offset * stride_out
    
    # Create offsets for the N dimension
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load in_0 and in_1, mask out of bounds
    mask = col_offsets < N
    
    # Load in_0 (position_bias) and in_1 (scores)
    in_0 = tl.load(in_0_row_ptr + col_offsets, mask=mask, other=float('-inf'))
    in_1 = tl.load(in_1_row_ptr + col_offsets, mask=mask, other=float('-inf'))
    
    # Perform in-place add: in_1 += in_0
    added = in_0 + in_1
    
    # Compute softmax over the last dimension
    # Subtract max for numerical stability
    max_val = tl.max(added, axis=0)
    shifted = added - max_val
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    softmax_result = exp_shifted / sum_exp
    
    # Store result
    tl.store(out_row_ptr + col_offsets, softmax_result, mask=mask)


@torch.fx.wrap
def fused_add_softmax_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused add + softmax kernel.
    Since in_1 is modified in-place in the original, we need to add in_0 to in_1
    and then apply softmax.
    """
    # Get shapes
    N = in_1.shape[-1]  # softmax dimension
    M = in_1.numel() // N  # all other dimensions flattened
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Choose block size based on N
    BLOCK_SIZE_N = min(1024, next_power_of_2(N))
    num_programs = M
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_add_softmax_kernel[grid](
        in_0, in_1, out,
        N, M,
        in_0.stride(0), in_1.stride(0), out.stride(0),
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern:
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    
    Note: dropout with training=False is a no-op (just returns input).
    The float() and type_as() are also no-ops for float32 tensors.
    """
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax_kernel_wrapper


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1