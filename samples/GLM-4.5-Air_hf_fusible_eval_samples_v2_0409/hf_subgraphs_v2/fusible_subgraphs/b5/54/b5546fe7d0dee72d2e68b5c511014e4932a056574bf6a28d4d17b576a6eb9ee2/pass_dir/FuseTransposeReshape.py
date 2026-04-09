import torch
import triton
import triton.language as tl

def pattern(tmp_9):
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 152)
    return tmp_11

@triton.jit
def transpose_reshape_kernel(
    input_ptr,    # tmp_9: [1, 8, 197, 19]
    output_ptr,   # tmp_11: [1, 197, 152]
    B, C_in, H_in, W_in,  # input dimensions
    C_out, H_out, W_out,  # output dimensions
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = B * C_out * H_out * W_out
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE_M)
    
    if pid >= num_programs:
        return
        
    block_start = pid * BLOCK_SIZE_M
    offset = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offset < total_elements
    
    # Calculate output coordinates: [1, 197, 152]
    offset_3d = offset.reshape((1, C_out, H_out))
    b, c_out, h_out = offset_3d[0, 0, 0], offset_3d[0, 0, 1], offset_3d[0, 0, 2]
    
    # Map to input coordinates: [1, 8, 197, 19]
    # We're essentially doing a transpose(1,2) + reshape:
    # Original: [1, 8, 197, 19] -> transpose(1,2) -> [1, 197, 8, 19] -> reshape -> [1, 197, 152]
    # So: c_out maps to input height, h_out maps to input channels, and w_out maps to input width
    c_in = h_out  # channels in input become height in output
    h_in = c_out  # height in input becomes channels in output  
    w_in = offset // (B * C_out * H_out)  # width position in flattened last two dimensions
    
    # Load input value and store to output
    if w_in < W_in:  # Boundary check
        input_offset = b * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in
        input_val = tl.load(input_ptr + input_offset, mask=mask)
        output_offset = b * C_out * H_out + c_out * H_out + h_out
        tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def fused_transpose_reshape(tmp_9):
    # Simple transpose + reshape using regular torch operations
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 152)
    return tmp_11

def replacement_args(tmp_9):
    return (tmp_9,)

def replacement_func():
    return fused_transpose_reshape