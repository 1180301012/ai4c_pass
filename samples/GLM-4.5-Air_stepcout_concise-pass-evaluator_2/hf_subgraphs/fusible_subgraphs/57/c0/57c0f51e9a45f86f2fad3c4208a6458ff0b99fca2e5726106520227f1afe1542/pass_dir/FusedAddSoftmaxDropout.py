import torch
import triton
import triton.language as tl


# Pattern matching function - match dropout with training=False (no-op)
def pattern(tmp_3):
    # Dropout with training=False is a no-op - returns input unchanged
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(tmp_3):
    return (tmp_3,)


# Optimized fused add-softmax kernel
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    output_ptr,
    stride_in_0: tl.constexpr,
    stride_in_1: tl.constexpr,
    stride_out: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs: output = softmax((in_1 + in_0), dim=-1)
    
    This fuses the addition and softmax into a single kernel for better memory
    access patterns and reduced kernel launch overhead.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this program
    # Each program handles one "row" along the softmax dimension
    row_offset = pid * N
    
    # Load in_0 and in_1 for this row
    # Using BLOCK_SIZE for the softmax dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary check
    mask = offsets < N
    
    # Load in_0 (position_bias) - shape [*, 16, *, *] 
    # We only need the part that corresponds to this row
    in_0_ptrs = in_0_ptr + row_offset + offsets
    in_0_vals = tl.load(in_0_ptrs, mask=mask, other=float('-inf'))
    
    # Load in_1 (scores)
    in_1_ptrs = in_1_ptr + row_offset + offsets
    in_1_vals = tl.load(in_1_ptrs, mask=mask, other=float('-inf'))
    
    # Fused addition: in_1 + in_0
    added_vals = in_0_vals + in_1_vals
    
    # Softmax computation
    # Step 1: find max for numerical stability
    max_vals = tl.max(added_vals, axis=0)
    
    # Step 2: subtract max and exp
    shifted_vals = added_vals - max_vals
    exp_vals = tl.exp(shifted_vals)
    
    # Step 3: sum for normalization
    sum_vals = tl.sum(exp_vals, axis=0)
    
    # Step 4: divide to get softmax
    softmax_vals = exp_vals / sum_vals
    
    # Store result
    output_ptrs = output_ptr + row_offset + offsets
    tl.store(output_ptrs, softmax_vals, mask=mask)


def fused_add_softmax_kernel_2d(
    in_0_ptr,
    in_1_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for 2D tensors: output = softmax((in_1 + in_0), dim=-1)
    Handles the full tensor with proper 2D tiling.
    """
    # Get program IDs for 2D grid
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Calculate the starting offset for this batch/head combination
    row_offset = pid_batch * N + pid_head * N * N  # Simplified for the specific shape
    
    # For now, use a simpler 1D-like approach with better memory coalescing
    pass


# Better optimized kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel_autotuned(
    in_0_ptr,
    in_1_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Autotuned fused kernel: output = softmax((in_1 + in_0), dim=-1)
    """
    # Get program ID - each program handles one softmax row
    pid = tl.program_id(0)
    
    # Calculate starting position
    row_start = pid * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load in_0 (additive bias)
    in_0_ptrs = in_0_ptr + row_start + offsets
    in_0_vals = tl.load(in_0_ptrs, mask=mask, other=float('-inf'))
    
    # Load in_1 (scores to be biased)
    in_1_ptrs = in_1_ptr + row_start + offsets
    in_1_vals = tl.load(in_1_ptrs, mask=mask, other=float('-inf'))
    
    # Fused addition
    added = in_0_vals + in_1_vals
    
    # Softmax with numerical stability
    # Max subtraction for numerical stability
    max_val = tl.max(added, axis=0)
    shifted = added - max_val
    exp_val = tl.exp(shifted)
    
    # Sum and normalize
    sum_val = tl.sum(exp_val, axis=0)
    softmax_out = exp_val / sum_val
    
    # Store
    out_ptrs = output_ptr + row_start + offsets
    tl.store(out_ptrs, softmax_out, mask=mask)


# Identity function wrapper - dropout with training=False returns input unchanged
@torch.fx.wrap
def kernel_wrapper(tmp_3):
    """
    Since dropout with training=False is a no-op (just returns the input),
    we can simply return the input directly without any computation.
    """
    return tmp_3


def replacement_func():
    return kernel_wrapper