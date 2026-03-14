import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete computation including dropout
def pattern(in_0, in_1):
    """
    Match the complete computation pattern from original model:
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    
    This matches the entire computation and returns tmp_2 which is observable.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel - fused SiLU * multiplication (identity for dropout)
@triton.jit
def fused_silu_multiply_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: silu(x) * y = (x * sigmoid(x)) * y
    This combines SiLU activation and multiplication in a single kernel,
    effectively eliminating the unnecessary dropout with p=0.0.
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: silu(x) * y = x * sigmoid(x) * y
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    result = silu_x * y
    
    # Store result (dropout with p=0.0 is effectively an identity operation)
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_silu_multiply_with_dropout_identity(x, y):
    """
    Fused SiLU + multiplication operation that computes:
    result = silu(x) * y = (x * sigmoid(x)) * y
    
    This eliminates both the intermediate memory allocation for the SiLU output
    and the unnecessary dropout operation with p=0.0 (which becomes an identity).
    """
    # For shapes like [1, 257, 1024], use optimized block size
    # Total elements = 1 * 257 * 1024 = 263,168
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimized for this tensor size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float32)
    
    fused_silu_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_silu_multiply_with_dropout_identity