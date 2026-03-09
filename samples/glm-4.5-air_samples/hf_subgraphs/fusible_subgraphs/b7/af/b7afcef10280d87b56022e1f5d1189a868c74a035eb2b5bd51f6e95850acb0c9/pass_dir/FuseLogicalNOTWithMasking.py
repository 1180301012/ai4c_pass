import torch
import triton
import triton.language as tl

@triton.jit
def logical_not_masked_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar_value: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for logical NOT + masked_fill
    Input: int64 values (0/1)
    Output: float32 where not(input) = scalar_value, and input=1 positions are fill_value
    """
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as int64 and convert to float32 immediately
    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    
    # Compute logical NOT: scalar_value - x, and directly apply masking
    result = tl.where(x == 1.0, fill_value, scalar_value - x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_logical_not_masked(input):
    """Fused implementation of logical NOT + masked_fill
    Args:
        input: int64 tensor with values 0 or 1
    Returns:
        float32 tensor where not(input) = 1, and input=1 positions are -inf
    """
    # Create output tensor with same shape as input but float32 dtype
    output = torch.empty(input.shape, dtype=torch.float32, device='cuda')
    
    # Handle different tensor shapes by flattening for better GPU utilization
    input_flat = input.flatten()
    output_flat = output.flatten()
    
    N = input_flat.numel()
    if N == 0:
        return output
    
    # Define the constants that match the original computation
    scalar_value = 1.0
    fill_value = -3.4028234663852886e+38
    
    BLOCK_SIZE = 1024  # Optimal for most GPUs
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    logical_not_masked_kernel[(num_programs,)](
        input_ptr=input_flat,
        output_ptr=output_flat,
        n_elements=N,
        scalar_value=scalar_value,
        fill_value=fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x):
    """Match the computation pattern: convert + arithmetic + boolean + masked_fill
    
    Args:
        x: The input tensor (in_0)
    Returns:
        The final output tensor after all operations
    """
    # Match the exact computation sequence from model.py
    tmp_0 = x.to(torch.float32)
    tmp_1 = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    tmp_2 = tmp_1 - tmp_0
    tmp_1 = None
    tmp_3 = tmp_2.to(torch.bool) 
    result = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
    tmp_2 = tmp_3 = None
    return result

def replacement_args(*args):
    """Extract arguments for the replacement function"""
    # args should contain (input_tensor) based on pattern
    return (args[0],)

def replacement_func():
    """Return the fused kernel function"""
    return fused_logical_not_masked