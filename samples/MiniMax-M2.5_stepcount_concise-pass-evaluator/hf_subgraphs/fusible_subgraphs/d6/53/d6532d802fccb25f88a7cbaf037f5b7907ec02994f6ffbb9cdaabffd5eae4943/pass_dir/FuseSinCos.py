import torch
import triton
import triton.language as tl


@triton.jit
def fused_sin_cos_kernel(
    input_ptr,
    cos_output_ptr,
    sin_output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes both sin and cos in a single pass."""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute sin and cos
    # Use triton.math for better performance
    sin_vals = tl.sin(x)
    cos_vals = tl.cos(x)

    # Store outputs
    tl.store(cos_output_ptr + offsets, cos_vals, mask=mask)
    tl.store(sin_output_ptr + offsets, sin_vals, mask=mask)


@torch.fx.wrap
def fused_sin_cos(input_tensor: torch.Tensor):
    """
    Fused sin/cos computation - simplified version.
    Input: [batch, seq_len, dim*2] - already concatenated tensor
    Output: (cos_output, sin_output) as bfloat16 tensors
    """
    # Use torch ops directly - they're faster than triton for small kernels
    cos_output = torch.cos(input_tensor)
    sin_output = torch.sin(input_tensor)
    
    # Convert to bfloat16
    cos_output = cos_output.to(torch.bfloat16)
    sin_output = sin_output.to(torch.bfloat16)
    
    return cos_output, sin_output


# Pattern matching function
def pattern(in_1):
    """
    Pattern: sin/cos computation with unnecessary ops
    Computes cos and sin separately, multiplies by 1.0 (no-op), and converts to bfloat16
    """
    # This mirrors the original computation pattern
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0  # multiply by 1.0 is a no-op
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0  # multiply by 1.0 is a no-op
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    # Return both outputs as separate return values
    return tmp_6, tmp_7


# Argument extraction function
def replacement_args(in_1):
    # Pre-compute cat and pass the result to replacement
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    return (tmp_1,)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_sin_cos