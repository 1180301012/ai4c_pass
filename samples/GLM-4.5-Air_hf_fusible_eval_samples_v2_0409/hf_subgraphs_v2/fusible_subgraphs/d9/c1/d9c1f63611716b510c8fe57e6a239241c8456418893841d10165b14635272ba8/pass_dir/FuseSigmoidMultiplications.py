import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_mul_kernel(
    sigmoid_input1_ptr,
    sigmoid_input2_ptr,
    mul_input1_ptr,
    mul_input2_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    sigmoid_input1 = tl.load(sigmoid_input1_ptr + offsets, mask=mask, other=0.0)
    sigmoid_input2 = tl.load(sigmoid_input2_ptr + offsets, mask=mask, other=0.0)
    mul_input1 = tl.load(mul_input1_ptr + offsets, mask=mask, other=0.0)
    mul_input2 = tl.load(mul_input2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid operations
    sigmoid1 = tl.sigmoid(sigmoid_input1.float())
    sigmoid2 = tl.sigmoid(sigmoid_input2.float())
    
    # Perform element-wise multiplications
    result1 = sigmoid1 * mul_input1
    result2 = sigmoid2 * mul_input2
    
    # Store results
    tl.store(out1_ptr + offsets, result1.to(sigmoid_input1.dtype), mask=mask)
    tl.store(out2_ptr + offsets, result2.to(sigmoid_input2.dtype), mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiplication(sigmoid_input1, sigmoid_input2, mul_input1, mul_input2):
    # Get total number of elements
    n_elements = sigmoid_input1.numel()
    
    # Handle different shapes - flatten for processing
    original_shapes = [sigmoid_input1.shape, sigmoid_input2.shape]
    
    # Flatten all tensors to 1D for processing
    sigmoid_input1_flat = sigmoid_input1.flatten()
    sigmoid_input2_flat = sigmoid_input2.flatten()
    mul_input1_flat = mul_input1.flatten()
    mul_input2_flat = mul_input2.flatten()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure all tensors are on the same device
    if mul_input1_flat.device != sigmoid_input1.device:
        mul_input1_flat = mul_input1_flat.to(sigmoid_input1.device)
        mul_input2_flat = mul_input2_flat.to(sigmoid_input1.device)
    
    # Launch the kernel
    fused_sigmoid_mul_kernel[(num_programs,)](
        sigmoid_input1_flat,
        sigmoid_input2_flat,
        mul_input1_flat,
        mul_input2_flat,
        sigmoid_input1_flat,  # Use same output pointers for in-place operations
        sigmoid_input2_flat,
        n_elements,
        BLOCK_SIZE
    )
    
    # Reshape back to original shapes
    result1 = sigmoid_input1_flat.view(original_shapes[0])
    result2 = sigmoid_input2_flat.view(original_shapes[1])
    
    return result1, result2

def pattern(tmp_9, in_9, tmp_12, tmp_13):
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    return tmp_11, tmp_15, tmp_16

def replacement_args(tmp_9, in_9, tmp_12, tmp_13):
    return (tmp_9, in_9, tmp_12.unsqueeze(-2), tmp_13)

def replacement_func():
    return fused_sigmoid_multiplication