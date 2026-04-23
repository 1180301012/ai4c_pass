import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_mul_kernel(
    x_ptr,
    excitation_ptr,
    output_ptr,
    n_elements,
    x_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: sigmoid(x) * excitation
    This combines:
    - sigmoid activation
    - element-wise multiplication with excitation (with broadcasting)
    All in a single kernel to minimize memory traffic.
    
    Handles broadcasting where x (sigmoid output) has smaller spatial dimensions.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate broadcasted value for each output position
    # x (sigmoid output) broadcasts to excitation shape
    x_idx = offsets % x_numel
    
    # Load sigmoid output - use broadcasting index
    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    # Load excitation input - use original index
    excitation = tl.load(excitation_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for computation (required for tl.exp which doesn't support bf16/fp16)
    # Then convert back to original dtype for storage
    x_fp32 = x.to(tl.float32)
    excitation_fp32 = excitation.to(tl.float32)
    
    # Compute sigmoid in fp32: sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
    
    # Multiply sigmoid by excitation (with broadcasting) in fp32
    mul_result = sigmoid_x * excitation_fp32
    
    # Convert back to original dtype and store
    # The original dtype comes from the excitation tensor
    result = mul_result.to(excitation.dtype)
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul(x, excitation):
    """
    Fused implementation of: sigmoid(x) * excitation
    Input shapes: x [B, C, H, W], excitation [B, C2, H2, W2] with broadcasting
    """
    n_elements = excitation.numel()
    x_numel = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(excitation)
    
    sigmoid_mul_kernel[(num_programs,)](
        x_ptr=x,
        excitation_ptr=excitation,
        output_ptr=output,
        n_elements=n_elements,
        x_numel=x_numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_2, in_3):
    """
    Match the pattern: sigmoid -> multiply
    This matches the SE module computation after conv2d.
    The sigmoid result is multiplied with excitation (in_2).
    Note: gelu is NOT included in the pattern - it's left in the original graph
    to maintain exact numerical equivalence with PyTorch's exact GELU formula.
    """
    tmp_3 = in_3.sigmoid()
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_2, in_3):
    """
    Extract arguments needed for the fused implementation.
    """
    return (in_2, in_3)


def replacement_func():
    """
    Returns the fused kernel that replaces the sigmoid-multiply pattern.
    GELU is applied separately in the original graph for correctness.
    """
    return fused_sigmoid_mul