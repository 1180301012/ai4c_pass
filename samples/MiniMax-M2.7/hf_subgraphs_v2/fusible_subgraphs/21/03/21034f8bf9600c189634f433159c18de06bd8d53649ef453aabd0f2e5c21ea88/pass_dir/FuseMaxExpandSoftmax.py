import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=4),
    ],
    key=["dim"],
)
@triton.jit
def fused_max_exp_softmax_kernel(
    in_ptr,
    out_ptr,
    num_rows: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. max reduction along last dimension
    2. expand (broadcast)
    3. subtract (for numerical stability)
    4. exp
    5. softmax sum reduction
    
    Each row (identified by program_id) processes one softmax row of length `dim`.
    """
    row_id = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_id * dim
    
    # Load all elements in this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + dim
    
    x = tl.load(in_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Step 1: Find max of this row (softmax uses max for stability)
    row_max = tl.max(x, axis=0)
    
    # Step 2: Subtract max for numerical stability and exp
    x_sub = x - row_max
    x_exp = tl.exp(x_sub)
    
    # Step 3: Compute sum of exponentials (for normalization)
    row_sum = tl.sum(x_exp, axis=0)
    
    # Step 4: Normalize to get softmax
    softmax_out = x_exp / row_sum
    
    # Store result
    tl.store(out_ptr + offsets, softmax_out, mask=mask)


def pattern(in_0, in_1):
    """
    Match the pattern:
    max_1 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = max_1[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = in_1.view(...)
    return (tmp_4, tmp_5)
    """
    max_1 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = max_1[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    
    # Extract the view size from in_1 shape
    # Pattern varies: (12, 512, -1), (1, 512, -1), (32, 512, -1), (8, 512, -1), (2, 512, -1)
    b, h = in_1.shape[0], in_1.shape[1]
    tmp_5 = in_1.view(b, h, -1)
    
    return (tmp_4, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def fused_softmax_wrapper(in_0, in_1):
    """
    Fused implementation of max + expand + subtract + softmax
    """
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    dim = in_0.shape[2]
    
    # Total number of rows to process (softmax is computed per row along last dim)
    num_rows = batch_size * seq_len
    
    # Create output tensor
    out_softmax = torch.empty_like(in_0)
    
    # Grid configuration - one program per row
    grid = (num_rows,)
    
    # Launch kernel
    fused_max_exp_softmax_kernel[grid](
        in_0,
        out_softmax,
        num_rows,
        dim,
    )
    
    # Handle the view of in_1
    b, h = in_1.shape[0], in_1.shape[1]
    tmp_5 = in_1.view(b, h, -1)
    
    return (out_softmax, tmp_5)


def replacement_func():
    return fused_softmax_wrapper