import torch
import triton
import triton.language as tl

# Pattern matching for element-wise add + relu fusion
# This fuses the residual connection with ReLU activation
def pattern(in_0, in_1):
    """
    Match the computation:
    relu(x + in_0) where x is the result of previous operations
    
    The pattern matches: tmp_3 += in_0 followed by relu
    """
    # Step 1: in-place add (residual connection)
    tmp_3 = in_1 + in_0
    
    # Step 2: ReLU activation (inplace)
    tmp_5 = torch.nn.functional.relu(tmp_3, inplace=True)
    
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_relu_kernel(
    in_1_ptr,      # The main tensor after SE gating
    in_0_ptr,      # The residual connection tensor  
    out_ptr,       # Output tensor
    N: tl.constexpr,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: relu(in_1 + in_0)
    
    This fuses the element-wise addition with ReLU activation.
    """
    # Calculate global thread offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load both tensors
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    
    # Compute add + relu: relu(x + residual) = max(0, x + residual)
    added = x + residual
    result = tl.where(added > 0, added, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_relu_wrapper(in_0, in_1):
    """
    Wrapper for the fused add + relu kernel.
    
    Args:
        in_0: residual connection tensor
        in_1: main tensor after SE gating
    
    Returns:
        ReLU(x + residual)
    """
    N = in_1.numel()
    
    # Allocate output
    out = torch.empty_like(in_1)
    
    # Calculate grid
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_relu_kernel[(num_programs,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_add_relu_wrapper