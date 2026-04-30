import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
# in_0: tensor of shape [1, 16, 196, 196]
# Pattern: sum(dim=-1) + unsqueeze(-1) + divide + dropout(p=0.0)
def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    in_0_div = in_0 / tmp_1
    tmp_3 = torch.nn.functional.dropout(in_0_div, 0.0, False, False)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for row-wise sum normalization
@triton.jit
def fused_sum_normalize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    n_rows,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute row start and end offsets
    row_offset = row_idx * row_size
    
    # Load all elements in the row and compute sum
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * row_size
    
    # For the reduction, we need multiple passes
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum using iterative reduction
    # First accumulate in registers using loop unrolling
    sum_val = tl.sum(x, axis=0)
    
    # Broadcast sum back to all elements and normalize
    # sum_val is scalar, we need to broadcast it
    normalized = x / sum_val
    
    # Store results
    tl.store(output_ptr + offsets, normalized, mask=mask)

# Autotuning configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['row_size'],
)
@triton.jit
def fused_sum_normalize_autotuned_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    n_rows,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute row start offsets
    row_offset = row_idx * row_size
    
    # Load all elements in the row
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * row_size
    
    # Load elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum across the row
    sum_val = tl.sum(x, axis=0)
    
    # Normalize (divide by sum)
    normalized = x / sum_val
    
    # Store results
    tl.store(output_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def fused_sum_normalize_wrapper(x):
    """
    Fused kernel that computes: x / x.sum(dim=-1, keepdim=True)
    Input shape: [batch, heads, seq_len, seq_len] = [1, 16, 196, 196]
    Output: normalized tensor of same shape
    """
    shape = x.shape
    batch_size = shape[0]
    num_heads = shape[1]
    seq_len = shape[2]
    row_size = shape[3]  # This is the dimension we sum over
    
    n_rows = batch_size * num_heads * seq_len  # Total number of rows to process
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Use autotuned kernel
    grid = (n_rows,)
    BLOCK_SIZE = 1024  # Will be auto-tuned
    
    fused_sum_normalize_autotuned_kernel[grid](
        x, output, n_elements, n_rows, row_size, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sum_normalize_wrapper