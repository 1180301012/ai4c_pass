import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = in_2.transpose(-1, -2)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    dim3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate total elements and block ranges
    total_elements = dim0 * dim1 * dim2 * dim3
    block_start = pid * BLOCK_SIZE
    block_end = min((pid + 1) * BLOCK_SIZE, total_elements)
    
    if block_start >= total_elements:
        return
    
    # Calculate linearized indices and reshape to original dimensions
    indices = tl.arange(block_start, block_end)
    i = indices // (dim1 * dim2 * dim3)
    j = (indices % (dim1 * dim2 * dim3)) // (dim2 * dim3)
    k = (indices % (dim2 * dim3)) // dim3
    l = indices % dim3
    
    # Transpose: swap last two dimensions (dim2 <-> dim3)
    k_new = l
    l_new = k
    
    # Calculate new linear indices
    new_indices = i * (dim1 * dim3 * dim2) + j * (dim3 * dim2) + k_new * dim3 + l_new
    
    # Load input and store output
    input_vals = tl.load(input_ptr + indices, mask=indices < total_elements, other=0.0)
    tl.store(output_ptr + new_indices, input_vals, mask=new_indices < total_elements)

@torch.fx.wrap
def optimized_transpose(in_0, in_1, in_2):
    # Handle transpose with Triton kernel
    original_shape = in_2.shape
    
    if len(original_shape) == 4:
        # 4D tensor: (dim0, dim1, dim2, dim3) -> transpose(-1, -2) -> (dim0, dim1, dim3, dim2)
        dim0, dim1, dim2, dim3 = original_shape
        
        # Create output tensor
        output_shape = (dim0, dim1, dim3, dim2)
        transposed = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
        
        # Set block size and calculate grid size
        BLOCK_SIZE = 1024
        grid_size = (dim0 * dim1 * dim2 * dim3 + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        transpose_kernel[(grid_size,)](
            in_2,
            transposed,
            dim0, dim1, dim2, dim3,
            BLOCK_SIZE
        )
    else:
        # For other dimensions, fall back to PyTorch transpose
        transposed = in_2.transpose(-1, -2)
    
    # For the reshape part, we'll fall back to the original for now
    # This pass focuses on transpose optimization
    matmul = torch.matmul(in_1, in_0)
    reshaped = torch.reshape(matmul, [-1, 16])
    
    return (reshaped, transposed)

def replacement_func():
    return optimized_transpose