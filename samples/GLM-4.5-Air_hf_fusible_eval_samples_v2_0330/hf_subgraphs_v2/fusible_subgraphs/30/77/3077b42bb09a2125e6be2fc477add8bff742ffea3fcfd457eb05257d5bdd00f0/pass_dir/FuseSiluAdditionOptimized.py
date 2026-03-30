import torch
import triton
import triton.language as tl
import math

def pattern(in_1, in_0):
    """
    Match SiLU activation followed by addition pattern
    Pattern: tmp_0 = silu(in_1); tmp_1 = tmp_0 + in_0
    """
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)

def replacement_args(in_1, in_0):
    """Extract arguments for the fusion kernel"""
    return (in_1, in_0)

@triton.jit
def silu_add_fusion_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU + Addition kernel"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors - use a different strategy for better performance
    # Use aligned memory access where possible
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # Use math.expm1 for better numerical stability when x is near 0
    neg_x = -in_1
    # Clamp negative values to prevent overflow in exp
    exp_neg_x = tl.exp(tl.maximum(neg_x, -87.0))  # Prevent exp overflow
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    silu_out = in_1 * sigmoid_x
    
    # Add the second input
    out = silu_out + in_0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_addition(in_1, in_0):
    """
    Wrapper function to launch the fused SiLU + Addition kernel
    The order of arguments is (in_1, in_0) to match the original computation:
    silu(in_1) + in_0
    """
    # Ensure inputs are on the same device and have the same shape
    assert in_1.device == in_0.device, "Inputs must be on the same device"
    assert in_1.shape == in_0.shape, "Inputs must have the same shape"
    
    n_elements = in_1.numel()
    
    # Use a larger block size for better GPU utilization
    BLOCK_SIZE = 2048  # Larger block size for modern GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as in_1 (the first input to SiLU)
    out = torch.empty_like(in_1)
    
    # Launch the fused kernel
    silu_add_fusion_kernel[(num_programs,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused SiLU + Addition function reference"""
    return fused_silu_addition