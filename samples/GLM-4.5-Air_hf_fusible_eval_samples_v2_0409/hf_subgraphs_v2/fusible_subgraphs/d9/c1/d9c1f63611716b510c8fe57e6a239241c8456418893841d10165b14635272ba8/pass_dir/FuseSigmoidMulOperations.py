import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_mul_kernel(
    sigmoid1_ptr,
    sigmoid2_ptr,
    mul1_ptr, 
    mul2_ptr,
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
    sigmoid1 = tl.load(sigmoid1_ptr + offsets, mask=mask, other=0.0)
    sigmoid2 = tl.load(sigmoid2_ptr + offsets, mask=mask, other=0.0)
    mul1 = tl.load(mul1_ptr + offsets, mask=mask, other=0.0)
    mul2 = tl.load(mul2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid operations using Triton's built-in sigmoid
    sigmoid1_result = tl.sigmoid(sigmoid1.float())
    sigmoid2_result = tl.sigmoid(sigmoid2.float())
    
    # Perform element-wise multiplications
    result1 = sigmoid1_result * mul1
    result2 = sigmoid2_result * mul2
    
    # Store results
    tl.store(out1_ptr + offsets, result1.to(sigmoid1.dtype), mask=mask)
    tl.store(out2_ptr + offsets, result2.to(sigmoid2.dtype), mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiplications(sigmoid_input1, sigmoid_input2, mul_input1, mul_input2):
    # Get total number of elements
    n_elements = sigmoid_input1.numel()
    
    # Ensure all tensors are on the same device and same dtype
    assert sigmoid_input1.dtype == sigmoid_input2.dtype, "Input dtypes must match"
    assert mul_input1.dtype == sigmoid_input1.dtype, "Mul1 dtype must match sigmoid1"
    assert mul_input2.dtype == sigmoid_input2.dtype, "Mul2 dtype must match sigmoid2"
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Make sure all tensors are contiguous for GPU memory access
    sigmoid_input1_contig = sigmoid_input1.contiguous()
    sigmoid_input2_contig = sigmoid_input2.contiguous()
    mul_input1_contig = mul_input1.contiguous()
    mul_input2_contig = mul_input2.contiguous()
    
    # Create output tensors
    result1 = torch.empty_like(sigmoid_input1_contig)
    result2 = torch.empty_like(sigmoid_input2_contig)
    
    # Launch the kernel
    fused_sigmoid_mul_kernel[(num_programs,)](
        sigmoid_input1_contig,
        sigmoid_input2_contig,
        mul_input1_contig,
        mul_input2_contig,
        result1,
        result2,
        n_elements,
        BLOCK_SIZE
    )
    
    return result1, result2

def pattern(tmp_9, in_9, tmp_14, tmp_13):
    tmp_11 = tmp_9.sigmoid()
    tmp_10 = in_9.sigmoid()
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    return tmp_15, tmp_16

def replacement_args(tmp_9, in_9, tmp_12, tmp_13):
    # Create tmp_14 by unsqueezing tmp_12 (this happens in the original computation)
    tmp_14 = tmp_12.unsqueeze(-2)
    return (tmp_9, in_9, tmp_14, tmp_13)

def replacement_func():
    return fused_sigmoid_multiplications