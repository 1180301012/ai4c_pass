import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Matches the pattern: ReLU(inplace=True) followed by Sigmoid
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Sigmoid kernel with optimized computation:
    - For x <= 0: result = sigmoid(0) = 0.5
    - For x > 0: result = 1 / (1 + exp(-x))
    However, since we want to maintain semantic equivalence with original:
    - ReLU first: max(0, x) 
    - Then Sigmoid: 1 / (1 + exp(-max(0, x)))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused ReLU + Sigmoid
    relu_out = tl.maximum(x, 0.0)
    
    # For sigmoid: use approximation for better performance
    # 1 / (1 + exp(-relu_out)) can be computed efficiently
    # We'll use a numerically stable approach
    z = relu_out
    if z >= 0:
        sigmoid_out = 1.0 / (1.0 + tl.exp(-z))
    else:
        sigmoid_out = tl.exp(z) / (1.0 + tl.exp(z))
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(input_tensor):
    """
    Wrapper function to launch the fused ReLU-Sigmoid kernel
    """
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_relu_sigmoid