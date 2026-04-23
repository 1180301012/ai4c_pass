import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Match the pattern: unsqueeze(1) followed by transpose(2, 3)
    Input shape: [1, 1024, 128]
    After unsqueeze(1): [1, 1, 1024, 128]
    After transpose(2, 3): [1, 1, 128, 1024]
    """
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using 1D grid with vectorized operations
@triton.jit
def unsqueeze_transpose_kernel(
    input_ptr,
    output_ptr,
    total_elements: tl.constexpr,
    in_dim1: tl.constexpr,
    in_dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for fused unsqueeze(1) + transpose(2,3).
    
    Input shape: [1, in_dim1, in_dim2] (3D tensor)
    Output shape: [1, 1, in_dim2, in_dim1] (4D tensor)
    
    For output[b, 0, d2, d1], we read input[b, d1, d2]
    """
    # 1D grid: each thread handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Each element maps to output index
    # output[b=0, 0, d2, d1]
    # d1 = (d2 * in_dim1 + d1) / in_dim1 = d2 (integer division)
    # Actually, we need to compute inverse mapping
    
    # input linear index = d1 * in_dim2 + d2
    # For output linear index k:
    # k = d2 * in_dim1 + d1
    # We need to find d1, d2 such that d1 * in_dim2 + d2 = input_linear
    # and k = output_linear
    # d1 = input_linear // in_dim2
    # d2 = input_linear % in_dim2
    # k = d2 * in_dim1 + d1 = (input_linear % in_dim2) * in_dim1 + (input_linear // in_dim2)
    
    # For input at index offsets[i]:
    # d1 = offsets[i] // in_dim2
    # d2 = offsets[i] % in_dim2
    # output_idx = d2 * in_dim1 + d1
    
    input_idx = offsets
    d1 = input_idx // in_dim2
    d2 = input_idx % in_dim2
    output_idx = d2 * in_dim1 + d1
    
    # Load from input
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Store to output
    tl.store(output_ptr + output_idx, x, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    """
    Wrapper function that launches the optimized Triton kernel.
    Input: [1, 1024, 128] -> Output: [1, 1, 128, 1024]
    """
    in_batch = in_0.shape[0]  # 1
    in_dim1 = in_0.shape[1]   # 1024
    in_dim2 = in_0.shape[2]   # 128
    
    total_elements = in_batch * in_dim1 * in_dim2  # 131072
    
    # Output shape: [1, 1, 128, 1024]
    output = torch.empty(
        (in_batch, 1, in_dim2, in_dim1),
        dtype=in_0.dtype,
        device=in_0.device
    )
    
    # Use 1D grid with large BLOCK_SIZE to minimize launch overhead
    BLOCK_SIZE = 4096
    num_programs = triton.cdiv(total_elements, BLOCK_SIZE)
    
    unsqueeze_transpose_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        total_elements=total_elements,
        in_dim1=in_dim1,
        in_dim2=in_dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return kernel_wrapper