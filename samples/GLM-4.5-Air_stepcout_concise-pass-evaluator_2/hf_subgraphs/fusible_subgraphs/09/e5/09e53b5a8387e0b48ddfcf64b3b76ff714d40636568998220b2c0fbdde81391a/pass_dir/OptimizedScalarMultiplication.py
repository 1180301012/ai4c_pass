import torch
import triton
import triton.language as tl

def pattern(scalar, tensor1, tensor2):
    """
    Pattern: scalar multiplication followed by addition with a third tensor
    This targets: (tensor1 * scalar) + tensor2
    """
    # More specific pattern focusing on scalar multiplication + tensor addition
    multiplied = tensor1 * scalar
    result = multiplied + tensor2
    return result.contiguous()

def replacement_args(scalar, tensor1, tensor2):
    """
    Extract arguments for the replacement function
    """
    return (scalar, tensor1, tensor2)

@triton.jit
def scalar_mul_add_kernel(
    scalar_val,
    tensor1_ptr,
    tensor2_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel: result = tensor1 * scalar + tensor2
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    tensor1_data = tl.load(tensor1_ptr + offsets, mask=mask, other=0.0)
    tensor2_data = tl.load(tensor2_ptr + offsets, mask=mask, other=0.0)
    
    # Perform computation: tensor1 * scalar + tensor2
    result = tensor1_data * scalar_val + tensor2_data
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def scalar_mul_add_operation(scalar, tensor1, tensor2):
    """
    Wrapper function for the optimized scalar multiplication and addition
    """
    # Determine the size using tensor1 as the reference
    N = tensor1.numel()
    block_size = 1024
    num_programs = (N + block_size - 1) // block_size
    
    out = torch.empty_like(tensor1)
    
    scalar_mul_add_kernel[(num_programs,)](
        scalar,        # scalar value
        tensor1,       # first tensor
        tensor2,       # second tensor
        out,           # output buffer
        N,             # number of elements
        block_size,    # block size
    )
    
    return out

def replacement_func():
    """
    Return the optimized operation function
    """
    return scalar_mul_add_operation