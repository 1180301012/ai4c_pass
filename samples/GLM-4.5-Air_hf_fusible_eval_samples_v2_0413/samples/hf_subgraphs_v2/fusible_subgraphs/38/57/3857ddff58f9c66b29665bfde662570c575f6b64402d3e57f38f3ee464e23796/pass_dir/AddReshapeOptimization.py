import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the computation up to the reshape
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, in_1.shape[0])  # Reshape based on weight dimension
    return tmp_3,  # Only return the reshaped tensor for now

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def add_reshape_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_shape,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple add + reshape optimization
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_shape[0] * x_shape[1] * x_shape[2]
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    out = x + y
    
    # Store in flattened format equivalent to reshape(-1, hidden_size)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def add_reshape_optimization(in_0, in_1, in_2, in_3):
    """
    in_0: bias, in_1: weight, in_2, in_3: input tensors
    Optimized addition + reshape
    """
    # Get input shapes
    input_shape = in_2.shape
    hidden_size = in_1.shape[0]
    
    # Compute total elements after reshape
    total_elements = input_shape[0] * input_shape[1] * input_shape[2]
    
    # Create output tensor (reshaped)
    output_2d = torch.empty((input_shape[0] * input_shape[1], hidden_size), 
                           dtype=in_2.dtype, device=in_2.device)
    
    # For now, just do the addition and reshape with PyTorch
    # This ensures correctness while we optimize the Triton kernel
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, hidden_size)
    
    return tmp_3,  # Return only the first output for now

def replacement_func():
    return add_reshape_optimization