import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match linear + add + ReLU fusion pattern"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = in_2 + tmp_2
    tmp_4 = tmp_3.relu_()
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_add_relu_kernel(
    bias_ptr,                    # bias vector [N] 
    weight_ptr,                  # weight matrix [N, N] 
    input_add_ptr,               # input to add [M, N]
    input_linear_ptr,            # input to linear [M, N]
    output_ptr,                  # output [M, N]
    M: tl.constexpr,             # batch size (1000)
    N: tl.constexpr              # feature dim (128)
):
    """Fused linear + add + ReLU kernel - working version"""
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Check if this row is within bounds
    if row_idx >= M:
        return
    
    # Load input row
    input_row = tl.load(input_linear_ptr + row_idx * N + tl.arange(0, N))
    
    # Initialize accumulator with bias
    acc = tl.load(bias_ptr + tl.arange(0, N))
    
    # Compute linear transformation using tl.sum for dot products
    # For each output feature dimension j
    for j in range(N):
        # Load j-th weight column
        weight_col = tl.load(weight_ptr + j * N + tl.arange(0, N))
        
        # Dot product: input_row • weight_col
        dot_product = tl.sum(input_row * weight_col)
        
        # Add to accumulator
        acc += dot_product
    
    # Load add input row
    add_row = tl.load(input_add_ptr + row_idx * N + tl.arange(0, N))
    
    # Add input and apply ReLU using tl.maximum
    result = tl.maximum(acc + add_row, 0.0)
    
    # Store output row
    tl.store(output_ptr + row_idx * N + tl.arange(0, N), result)

@torch.fx.wrap
def fused_linear_add_relu(bias, weight, input_add, input_linear):
    """Wrapper function for the fused kernel"""
    # Get input shapes
    M, N = input_linear.shape
    
    # Get device
    device = input_linear.device
    
    # Create output tensor
    output = torch.empty((M, N), dtype=torch.float32, device=device)
    
    # Launch one program per row (simpler approach)
    fused_linear_add_relu_kernel[(M,)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_add_ptr=input_add,
        input_linear_ptr=input_linear,
        output_ptr=output,
        M=M,
        N=N
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_linear_add_relu