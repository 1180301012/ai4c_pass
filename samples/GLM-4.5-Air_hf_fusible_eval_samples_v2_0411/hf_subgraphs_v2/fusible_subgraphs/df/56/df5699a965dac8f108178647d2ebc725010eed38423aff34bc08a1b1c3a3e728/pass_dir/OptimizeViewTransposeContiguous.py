import torch
import triton
import triton.language as tl

# Pattern matching for view + transpose + contiguous + view optimization
def pattern(input_5d_tensor):
    # Pattern matching the exact sequence from model:
    # tmp_7 = tmp_5.view(N, 2, C, H, W)
    # tmp_8 = torch.transpose(tmp_7, 1, 2)  
    # tmp_9 = tmp_8.contiguous()
    # tmp_10 = tmp_9.view(N, C*2, H, W)
    transposed = torch.transpose(input_5d_tensor, 1, 2)
    contiguous_tensor = transposed.contiguous()
    # The final view merges the second and third dimensions: (N, C, 2, H, W) -> (N, C*2, H, W)
    reshaped_4d = contiguous_tensor.view(contiguous_tensor.shape[0], 
                                        contiguous_tensor.shape[1] * contiguous_tensor.shape[2],
                                        contiguous_tensor.shape[3], 
                                        contiguous_tensor.shape[4])
    return reshaped_4d

# Argument extraction function
def replacement_args(input_5d_tensor):
    return (input_5d_tensor,)

# Optimized Triton kernel for direct 5D to 4D reshape
@triton.jit
def direct_5d_to_4d_reshape_kernel(
    input_ptr, output_ptr,
    N, orig_C_2, orig_C_1, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_offset = pid % N
    c_offset = (pid // N) % (orig_C_2 * orig_C_1)
    h_offset = ((pid // (N * (orig_C_2 * orig_C_1))) % H)
    w_offset = (pid // (N * (orig_C_2 * orig_C_1) * H))
    
    if h_offset >= H or w_offset >= W:
        return
    
    # Calculate original 5D coordinates
    orig_c1_offset = c_offset % orig_C_1
    orig_c2_offset = c_offset // orig_C_1
    
    # Load from 5D layout: [N, 2, C_1, H, W]
    input_offset = (n_offset * 2 * orig_C_1 * H * W + 
                   orig_c2_offset * orig_C_1 * H * W + 
                   orig_c1_offset * H * W + 
                   h_offset * W + 
                   w_offset)
    
    input_val = tl.load(input_ptr + input_offset)
    
    # Store to 4D layout: [N, 2*C_1, H, W]
    output_offset = (n_offset * (orig_C_2 * orig_C_1) * H * W + 
                    c_offset * H * W + 
                    h_offset * W + 
                    w_offset)
    
    tl.store(output_ptr + output_offset, input_val)

# Wrapper function for the optimized reshape
@torch.fx.wrap
def optimized_5d_to_4d_reshape(input_5d_tensor):
    # Get input tensor shape
    N, dim2, orig_C_1, H, W = input_5d_tensor.shape
    
    # Calculate the output 4D shape
    orig_C_2 = dim2  # This is the "2" dimension in the original pattern
    C_4d = orig_C_2 * orig_C_1
    output_4d_shape = (N, C_4d, H, W)
    
    # Create output tensor
    output_4d = torch.empty(output_4d_shape, dtype=input_5d_tensor.dtype, device=input_5d_tensor.device)
    
    # Calculate grid size based on total elements in output
    total_elements = N * C_4d * H * W
    BLOCK_SIZE = 1024
    grid_size = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    direct_5d_to_4d_reshape_kernel[grid_size](
        input_5d_tensor, output_4d,
        N, orig_C_2, orig_C_1, H, W,
        BLOCK_SIZE
    )
    
    return output_4d

# Replacement function (returns the optimized function reference)
def replacement_func():
    return optimized_5d_to_4d_reshape