import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Direct expansion operation
    # Match: input_tensor[None, None, slice(None, None, None)]
    expanded = input_tensor[None, None, slice(None, None, None)]
    return expanded

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_expand_kernel(
    input_ptr,
    output_ptr,
    input_dim0,
    input_dim1,
    output_dim0,
    output_dim1,
    output_dim2,
    output_dim3,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element  
    output_idx = tl.program_id(0)
    
    # Calculate 4D coordinates
    d0, d1_flat = output_idx // (output_dim1 * output_dim2 * output_dim3), output_idx % (output_dim1 * output_dim2 * output_dim3)
    d1, d2_flat = d1_flat // (output_dim2 * output_dim3), d1_flat % (output_dim2 * output_dim3)
    d2, d3 = d2_flat // output_dim3, d2_flat % output_dim3
    
    # For the expansion [2, 128] -> [1, 1, 2, 128]
    # Only positions where d0=0, d1=0 should have values, others should be 0
    if d0 == 0 and d1 == 0 and d2 < input_dim0 and d3 < input_dim1:
        # Map to input tensor: output[0,0,d2,d3] = input[d2,d3]
        input_offset = d2 * input_dim1 + d3
        
        # Calculate output offset for this 4D position
        output_offset = d0 * (output_dim1 * output_dim2 * output_dim3) + \
                       d1 * (output_dim2 * output_dim3) + \
                       d2 * output_dim3 + d3
        
        # Load from input and store to output
        input_value = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + output_offset, input_value)
    else:
        # For other positions, store 0.0
        if output_idx < output_dim0 * output_dim1 * output_dim2 * output_dim3:
            tl.store(output_ptr + output_idx, 0.0)

@torch.fx.wrap
def optimized_expand(input_tensor):
    # Input shape: [2, 128]
    input_dim0, input_dim1 = input_tensor.shape
    
    # Expanded shape: [1, 1, 2, 128] 
    output_dim0, output_dim1, output_dim2, output_dim3 = 1, 1, input_dim0, input_dim1
    output_shape = (output_dim0, output_dim1, output_dim2, output_dim3)
    
    # Create output tensor
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    # Calculate number of elements and grid size
    total_elements = output_dim0 * output_dim1 * output_dim2 * output_dim3
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_expand_kernel[(num_programs,)](
        input_tensor,
        output_tensor,
        input_dim0, input_dim1,
        output_dim0, output_dim1, output_dim2, output_dim3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return optimized_expand