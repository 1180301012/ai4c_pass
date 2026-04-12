import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, broadcast_sigmoid):
    tmp_3 = in_1 * broadcast_sigmoid;  in_1 = broadcast_sigmoid = None
    tmp_3 += in_0;  tmp_4 = tmp_3;  tmp_3 = in_0 = None
    return tmp_4

def replacement_args(in_0, in_1, broadcast_sigmoid):
    return (in_0, in_1, broadcast_sigmoid)

@triton.jit
def elementwise_add_mul_kernel(
    in0_ptr,
    in1_ptr,
    sigmoid_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input tensors
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    sigmoid = tl.load(sigmoid_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: out = in1 * sigmoid + in0
    out = in1 * sigmoid + in0
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_elementwise_add_mul(in_0, in_1, broadcast_sigmoid):
    # Calculate total number of elements
    n_elements = in_1.numel()  # Use in_1 as reference since they have same shape
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_1)  # Same shape and dtype as in_1
    
    # Launch kernel
    elementwise_add_mul_kernel[(num_programs, )](
        in0_ptr=in_0,
        in1_ptr=in_1,
        sigmoid_ptr=broadcast_sigmoid,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_elementwise_add_mul