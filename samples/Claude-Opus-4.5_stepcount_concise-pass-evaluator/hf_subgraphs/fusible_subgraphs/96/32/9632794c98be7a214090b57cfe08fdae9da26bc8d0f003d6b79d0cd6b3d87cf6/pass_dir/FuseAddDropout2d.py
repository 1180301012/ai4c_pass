import torch
import triton
import triton.language as tl

# Pattern matching function - matches add followed by dropout2d with training=False
def pattern(in_3, in_4):
    """
    Match the pattern: add followed by dropout2d
    Since dropout2d with training=False is a no-op, we can optimize this
    """
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_3, in_4):
    return (in_3, in_4)

# Triton kernel for optimized add operation - optimized for large tensors
@triton.jit
def add_kernel_fast(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute addition (dropout2d with training=False is a no-op)
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_add_dropout(in_3, in_4):
    """
    Optimized add operation that replaces add + dropout2d (training=False)
    Since dropout2d with training=False just returns input unchanged,
    we can skip it entirely and just do the add.
    """
    # Since dropout with training=False is a no-op, just use torch.add
    # which is already highly optimized for GPU
    return in_4 + in_3

# Replacement function - returns the function reference (not a call)
def replacement_func():
    return triton_add_dropout