import torch
import triton
import triton.language as tl

@triton.jit
def optimized_tensor_ops_kernel(
    x_ptr, codewords_ptr, output_ptr,
    x_dims, codewords_dims,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for 1D grid
    pid = tl.program_id(0)
    
    # Calculate offset
    offset = pid * BLOCK_SIZE
    
    # Create expanded x directly: [1, 4096, 512] -> [1, 4096, 32, 512]
    # Pattern: repeat along dimension 2 (32 times)
    x_offset = offset // (x_dims[1] * x_dims[2])  # x_idx
    codeword_offset = (offset // x_dims[1]) % codewords_dims[0]  # codeword_idx
    feature_offset = offset % x_dims[2]  # feature_idx
    
    # Load x element [1, 4096, 512] -> treat as [4096, 512]
    x_val = tl.load(x_ptr + (x_offset * x_dims[2] + feature_offset),
                   mask=(x_offset < x_dims[0]),
                   other=0.0)
    
    # Load codeword element [32, 512] 
    codeword_val = tl.load(codewords_ptr + (codeword_offset * codewords_dims[1] + feature_offset),
                          mask=(codeword_offset < codewords_dims[0]),
                          other=0.0)
    
    # Compute expanded shape: x - codewords
    # The expanded x is effectively x repeated along dim=2
    result_val = x_val - codeword_val
    
    # Store result directly in expanded shape [1, 4096, 32, 512] -> treat as [4096 * 32, 512]
    expanded_offset = x_offset * codewords_dims[0] * codewords_dims[1] + codeword_offset * codewords_dims[1] + feature_offset
    tl.store(output_ptr + expanded_offset, result_val,
             mask=(x_offset < x_dims[0]) & (codeword_offset < codewords_dims[0]) & (feature_offset < codewords_dims[1]))

@torch.fx.wrap
def optimized_tensor_ops(x, codewords, softmax_result):
    """Optimized version that avoids intermediate unsqueeze/expand operations"""
    # x: [1, 4096, 512], codewords: [32, 512], softmax_result: [1, 4096, 32]
    expanded_shape = [1, 4096, 32, 512]
    output = torch.empty(expanded_shape, dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    BLOCK_SIZE = 512  # Process full feature vectors at once
    
    total_elements = expanded_shape[1] * expanded_shape[2] * expanded_shape[3]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_tensor_ops_kernel[(num_programs,)](
        x, codewords, output,
        [expanded_shape[1], expanded_shape[3]],  # x dims [4096, 512]
        [expanded_shape[2], expanded_shape[3]],  # codewords dims [32, 512]
        BLOCK_SIZE
    )
    
    # Expand softmax_result from [1, 4096, 32] to [1, 4096, 32, 1]
    unsqueezed_softmax = softmax_result.unsqueeze(3)
    
    return output, unsqueezed_softmax

def pattern(in_0, softmax_out, in_4):
    """Pattern to match optimized tensor operations: view + unsqueeze + expand -> direct computation"""
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = softmax_out.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return tmp_10, tmp_9

def replacement_args(in_0, softmax_out, in_4):
    return (in_0, softmax_out, in_4)

def replacement_func():
    return optimized_tensor_ops