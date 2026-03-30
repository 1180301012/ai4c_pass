import torch
import triton
import triton.language as tl

def linear(input, weight, bias):
    """Linear transformation: input @ weight.T + bias"""
    # input: [batch, seq_len, in_features]
    # weight: [out_features, in_features] 
    # bias: [out_features]
    # output: [batch, seq_len, out_features]
    return input @ weight.T + bias

def pattern(x):
    """Pattern matching: Simple transpose operation with specific dim parameters"""
    return x.transpose(1, 2)

def replacement_args(x):
    """Extract arguments for the fused kernel"""
    return (x,)

@triton.jit
def simple_transpose_kernel(
    input_ptr, output_ptr,
    dim0, dim1, dim2,
    BLOCK_SIZE: tl.constexpr
):
    """Simple kernel to transpose dims 1 and 2"""
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total elements
    total_elements = dim0 * dim1 * dim2
    
    # Create mask for bounds checking
    mask = pid < total_elements
    
    # Convert linear index to tensor coordinates
    offsets = pid
    i = offsets // (dim1 * dim2)
    remaining = offsets % (dim1 * dim2)
    j = remaining // dim2
    k = remaining % dim2
    
    # Transpose j and k
    j_transposed = k
    k_transposed = j
    
    # Convert back to linear index (transposed)
    new_offsets = i * (dim1 * dim2) + j_transposed * dim2 + k_transposed
    
    # Load and store together
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + new_offsets, val, mask=mask)

@torch.fx.wrap
def simple_transpose(x):
    """Simple transpose wrapper that swaps dims 1 and 2"""
    new_shape = (x.shape[0], x.shape[2], x.shape[1])
    output = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    
    total_elements = x.numel()
    grid_size = (total_elements + 1023) // 1024
    
    simple_transpose_kernel[(grid_size,)](
        input_ptr=x,
        output_ptr=output,
        dim0=x.shape[0],
        dim1=x.shape[1], 
        dim2=x.shape[2],
        BLOCK_SIZE=1024
    )
    
    return output

def replacement_func():
    """Return the simple transpose function"""
    return simple_transpose