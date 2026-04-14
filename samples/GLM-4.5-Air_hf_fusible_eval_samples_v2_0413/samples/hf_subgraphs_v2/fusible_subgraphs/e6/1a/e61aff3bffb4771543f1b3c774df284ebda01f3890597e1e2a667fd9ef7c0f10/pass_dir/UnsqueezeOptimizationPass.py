import torch
import triton
import triton.language as tl

def pattern(softmax_result):
    """
    Pattern matching for the final unsqueeze operation.
    This matches: softmax_result.unsqueeze(-1)
    
    This is the last operation in all computation graphs, returning the final result.
    """
    result = softmax_result.unsqueeze(-1)
    return result

def replacement_args(softmax_result):
    """Extract arguments needed for the optimized kernel"""
    return (softmax_result,)

@triton.jit
def optimized_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel that performs unsqueeze(-1) operation.
    
    This kernel efficiently adds a final dimension of size 1 to the tensor,
    changing shape from [batch_size, dim1, dim2] to [batch_size, dim1, dim2, 1].
    """
    total_input_elements = batch_size * dim1 * dim2
    
    for i in range(tl.program_id(0), total_input_elements, tl.num_programs(0)):
        # Load input value
        input_val = tl.load(input_ptr + i)
        
        # Store value and placeholder for the added dimension
        # Output layout: each input element is followed by a 0.0
        tl.store(output_ptr + i * 2, input_val)
        tl.store(output_ptr + i * 2 + 1, 0.0)

@torch.fx.wrap
def optimized_unsqueeze(softmax_result):
    """
    Wrapper function that executes optimized unsqueeze operation.
    
    This function efficiently adds a final dimension of size 1 to the tensor,
    changing shape from [batch_size, dim1, dim2] to [batch_size, dim1, dim2, 1].
    """
    if softmax_result.dim() != 3:
        # Fallback to original implementation for non-3D tensors
        return softmax_result.unsqueeze(-1)
    
    batch_size, dim1, dim2 = softmax_result.shape
    
    # Output shape: [batch_size, dim1, dim2, 1]
    output_shape = (batch_size, dim1, dim2, 1)
    output = torch.empty(output_shape, dtype=softmax_result.dtype, device=softmax_result.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * dim1 * dim2
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Run optimized kernel
    optimized_unsqueeze_kernel[(num_programs,)](
        input_ptr=softmax_result,
        output_ptr=output,
        batch_size=batch_size,
        dim1=dim1,
        dim2=dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the optimized kernel function"""
    return optimized_unsqueeze