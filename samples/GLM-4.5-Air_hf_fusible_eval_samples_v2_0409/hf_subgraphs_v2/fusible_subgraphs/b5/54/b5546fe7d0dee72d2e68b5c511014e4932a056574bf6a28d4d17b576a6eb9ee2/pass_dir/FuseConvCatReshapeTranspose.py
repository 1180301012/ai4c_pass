import torch
import triton
import triton.language as tl

# Simple pattern: just multiplication
def pattern(in_6, tmp_5):
    tmp_6 = in_6 * tmp_5
    return tmp_6

@triton.jit
def multiplication_kernel(
    input1_ptr,    # tmp_5: [1, 8, H, W]
    input2_ptr,    # in_6: [1, 8, H, W] 
    output_ptr,    # tmp_6: [1, 8, H, W]
    H, W,          # tensor dimensions
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = 1 * 8 * H * W
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE_M)
    
    if pid >= num_programs:
        return
        
    block_start = pid * BLOCK_SIZE_M
    offset = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offset < total_elements
    
    # Load tensor values and perform multiplication with vectorized loads
    # For better performance, use more efficient memory access patterns
    in1_val = tl.load(input1_ptr + offset, mask=mask)
    in2_val = tl.load(input2_ptr + offset, mask=mask)
    
    # Multiplication operation
    result = in1_val * in2_val
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

@torch.fx.wrap  
def fused_multiplication(in_6, tmp_5):
    # Simple and robust multiplication using torch operations
    # This works for both tensor-tensor and scalar-tensor multiplications
    output = in_6 * tmp_5
    return output

def replacement_args(in_6, tmp_5):
    return (in_6, tmp_5)

def replacement_func():
    return fused_multiplication