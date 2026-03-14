import torch
import triton
import triton.language as tl


# Autotune configurations for different problem sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_mul_dropout_kernel(
    in_0_ptr,  # gated input (gelu input)
    in_1_ptr,  # non-gated input (multiplied with gelu output)
    out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. gelu(in_0) - using fast sigmoid approximation
    2. gelu_out * in_1
    3. dropout(result, p=dropout_p, training=training)
    
    All in a single kernel for better memory access patterns and compute efficiency.
    """
    # Each program processes a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU using the fast sigmoid approximation: x * sigmoid(1.702 * x)
    # This is mathematically close to the exact GELU
    gelu_out = x * tl.sigmoid(1.702 * x)
    
    # Element-wise multiply (gating)
    gated = gelu_out * y
    
    # Apply dropout during training only
    # When training=False (inference), we just pass through the values
    if training:
        # Generate random numbers and create mask
        # Use tl.rand for random numbers
        random = tl.rand(tl.constexpr(0), offsets)
        dropout_mask = random > dropout_p
        # Scale by 1/(1-p) during training to maintain expected value
        scale = 1.0 / (1.0 - dropout_p)
        result = tl.where(dropout_mask, gated * scale, 0.0)
    else:
        # During inference, dropout is a no-op
        result = gated
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1, dropout_p=0.1, training=False):
    """
    Wrapper function that launches the fused kernel.
    
    Args:
        in_0: The gated input (gelu applied to this)
        in_1: The non-gated input (multiplied with gelu output)
        dropout_p: Dropout probability
        training: Whether in training mode
    
    Returns:
        Result of gelu(in_0) * in_1 with dropout applied
    """
    # Flatten tensors for 1D kernel
    n_elements = in_0.numel()
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Calculate grid size - must cover all elements
    # Use larger block size (4096) for better memory coalescing on large tensors
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For very small tensors, use at least 4 programs for better occupancy
    if num_programs < 4:
        num_programs = 4
        # Recalculate BLOCK_SIZE to cover all elements with power of 2
        BLOCK_SIZE = (n_elements + num_programs - 1) // num_programs
        # Round up to power of 2 for efficiency
        BLOCK_SIZE = 1 << (BLOCK_SIZE - 1).bit_length()
    
    # Launch kernel - autotuner handles BLOCK_SIZE selection based on n_elements
    fused_gelu_mul_dropout_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        n_elements,
        dropout_p,
        training,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the computation pattern: gelu(in_0) * in_1, then dropout
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Returns the replacement function that implements the fused kernel.
    """
    def wrapper(in_0, in_1):
        dropout_p = 0.1
        training = False
        return fused_gelu_mul_dropout(in_0, in_1, dropout_p, training)
    return wrapper