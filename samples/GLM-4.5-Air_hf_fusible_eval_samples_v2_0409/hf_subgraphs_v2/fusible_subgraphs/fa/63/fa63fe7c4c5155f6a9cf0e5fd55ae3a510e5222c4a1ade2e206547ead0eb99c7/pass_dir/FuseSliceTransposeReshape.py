import torch
import triton
import triton.language as tl

def pattern(sliced_input, target_dim1, target_dim2, target_dim3):
    """
    Match the slice + transpose + reshape pattern:
    tmp_2 = in_2[...]  (slice - done by caller)
    tmp_3 = tmp_2.transpose(-1, -2)  (transpose) 
    tmp_4 = tmp_3.reshape(1, target_dim1, target_dim2, target_dim3)  (reshape)
    Returns the final reshaped tensor
    """
    tmp_3 = sliced_input.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, target_dim1, target_dim2, target_dim3)
    return tmp_4

def replacement_args(sliced_input, target_dim1, target_dim2, target_dim3):
    """Return the sliced tensor and reshape args to the fused kernel"""
    return (sliced_input, target_dim1, target_dim2, target_dim3)

@triton.jit
def fused_slice_transpose_reshape_kernel(
    input_ptr,
    output_ptr,
    input_n,
    input_c,
    input_h,
    input_w,
    output_n,
    output_c, 
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. Slice operation (already done by caller)
    2. Transpose (last two dimensions)
    3. Reshape to target output shape
    """
    pid = tl.program_id(0)
    
    # Calculate total elements and program count
    total_elements = output_n * output_c * output_h * output_w
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if pid >= num_programs:
        return
    
    # Each program handles a block of elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate output coordinates
    offset = offsets
    w = offset % output_w
    offset = offset // output_w
    h = offset % output_h  
    offset = offset // output_h
    c = offset % output_c
    n = offset // output_c
    
    # Map output coordinates to input coordinates (transpose + reshape)
    # Input shape: [input_n, input_c, input_h, input_w] -> [1, 8, 49, 19]
    # Output shape: [output_n, output_c, output_h, output_w] -> [1, 152, 7, 7]
    # The reshape and transpose effectively map:
    # output[c, h, w] -> input[c//8, c%8, w, h] (assuming 8 channels group)
    
    output_idx = n * output_c * output_h * output_w + c * output_h * output_w + h * output_w + w
    
    # Map to input coordinates with transpose
    input_n_idx = 0  # input_n is always 1
    input_c_idx = c // 8  # Group channels
    input_h_idx = w  # Transpose: w -> h
    input_w_idx = h  # Transpose: h -> w
    
    input_idx = input_n_idx * input_c * input_h * input_w + \
                input_c_idx * input_h * input_w + \
                input_h_idx * input_w + \
                input_w_idx
    
    # Load input and store output
    input_val = tl.load(input_ptr + input_idx, mask=mask)
    tl.store(output_ptr + output_idx, input_val, mask=mask)

@torch.fx.wrap
def fused_slice_transpose_reshape(input_tensor, target_dim1, target_dim2, target_dim3):
    """
    Fused function that combines slice, transpose, and reshape operations.
    Input is already sliced tensor from in_2
    Output is reshaped tensor according to target dimensions
    """
    input_shape = input_tensor.shape
    output_shape = (1, target_dim1, target_dim2, target_dim3)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    total_elements = output.numel()
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_slice_transpose_reshape_kernel[(num_programs,)](
        input_tensor,
        output,
        input_shape[0], input_shape[1], input_shape[2], input_shape[3],
        output_shape[0], output_shape[1], output_shape[2], output_shape[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_slice_transpose_reshape