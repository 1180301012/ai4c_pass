import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern targeting the specific computation: scalar multiplication followed by addition
    This focuses on the latter part of the original computation: (result * in_0) + in_3
    """
    # Pattern matches: multiplication by scalar, then addition, then contiguous
    tmp_1 = in_2  # in_2 contains the result of (in_2 + in_1) from previous computation
    tmp_2 = tmp_1 * in_0  # multiplication by scalar in_0
    tmp_3 = tmp_2 + in_3  # addition with in_3
    tmp_4 = tmp_3.contiguous()  # make contiguous
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract all arguments for the replacement function
    """
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_kernel(
    scalar_val,
    tensor_ptr,
    added_tensor_ptr,
    out_ptr,
    n_elements: tl.constexpr,
):
    """
    Optimized kernel for: result = tensor * scalar + added_tensor
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load data
    tensor_data = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    added_data = tl.load(added_tensor_ptr + offsets, mask=mask, other=0.0)
    
    # Perform computation: tensor * scalar + added_tensor
    result = tensor_data * scalar_val + added_data
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_operation(scalar, tensor1, tensor2, tensor3):
    """
    Wrapper function for the optimized computation
    """
    # Use the right tensors for the computation
    # tensor1 is in_0 (scalar), tensor2 is in_2 (main tensor), tensor3 is in_3 (added tensor)
    # Note: in_1 is not used in this specific pattern
    
    N = tensor2.numel()  # Use tensor2's size since it's the main tensor
    block_size = 1024
    num_programs = (N + block_size - 1) // block_size
    
    out = torch.empty_like(tensor2)
    
    optimized_kernel[(num_programs,)](
        scalar,           # scalar value from in_0
        tensor2,          # main tensor from in_2
        tensor3,          # added tensor from in_3
        out,              # output buffer
        N,                # number of elements
    )
    
    return out

def replacement_func():
    """
    Return the optimized operation function
    """
    return optimized_operation