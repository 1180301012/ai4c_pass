import torch
import triton
import triton.language as tl

def sigmoid(x):
    return tl.where(x >= 0, 1 / (1 + tl.exp(-x)), tl.exp(x) / (1 + tl.exp(x)))

@triton.jit
def fused_sigmoid_view_expand_kernel(
    sigmoid_input_ptr,
    x1_ptr, 
    x0_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for bounds checking
    sigmoid_mask = offsets < N  # N = C = 2048
    
    # Load sigmoid input (shape [1, 1, C] -> [C])
    sigmoid_input = tl.load(sigmoid_input_ptr + offsets, mask=sigmoid_mask, other=0.0)
    
    # Compute sigmoid directly (optimized view: [1, 1, C] -> [1, C, 1, 1])
    sigmoid_output = sigmoid(sigmoid_input)
    
    # Load input tensors for element-wise operations
    x1_mask = offsets < (C * H * W)
    x1_elements = tl.load(x1_ptr + offsets, mask=x1_mask, other=0.0)
    x0_elements = tl.load(x0_ptr + offsets, mask=x1_mask, other=0.0)
    
    # Fused operations: sigmoid * x1 + x0
    # For elements with C <= offset < C*H*W, use the expanded sigmoid values
    # For s < C: sigmoid_output[s] * x1_elements[s] + x0_elements[s]
    # For s >= C: use 0.0 * x1_elements[s] + x0_elements[s] = x0_elements[s]
    expanded_sigmoid = tl.where(offsets < C, sigmoid_output, 1.0)
    
    fused_result = expanded_sigmoid * x1_elements + x0_elements
    
    # Store result
    tl.store(output_ptr + offsets, fused_result, mask=x1_mask)

@torch.fx.wrap
def fused_sigmoid_view_expand(in_2, in_1, in_0):
    C = in_1.shape[1]  # 2048 channels
    H = in_1.shape[2]
    W = in_1.shape[3]
    total_elements = C * H * W
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(in_1)
    
    fused_sigmoid_view_expand_kernel[(num_programs,)](
        sigmoid_input_ptr=in_2,
        x1_ptr=in_1,
        x0_ptr=in_0,
        output_ptr=output,
        N=C,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(in_2, in_1, in_0):
    """Match the pattern: sigmoid -> view -> expand -> multiply -> add"""
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 += in_0
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    return fused_sigmoid_view_expand