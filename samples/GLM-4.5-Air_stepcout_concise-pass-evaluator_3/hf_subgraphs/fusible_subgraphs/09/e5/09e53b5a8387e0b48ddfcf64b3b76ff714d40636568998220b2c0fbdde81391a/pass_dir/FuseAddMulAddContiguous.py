import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_0, in_3):
    # Pattern matches: tmp_1 * tmp_0 + in_3 followed by contiguous()
    # Here tmp_1 should already contain (in_2 + in_1) from previous operations
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 + in_3
    tmp_4 = tmp_3.contiguous()
    return tmp_4

def replacement_args(tmp_1, tmp_0, in_3):
    return (tmp_1, tmp_0, in_3)

@triton.jit
def fused_kernel(
    in_1_ptr, 
    in_3_ptr, 
    out_ptr,
    in_0_val,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load inputs - in_1_ptr already contains (in_2 + in_1) result
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse operations: (in_2 + in_1) * in_0 + in_3
    # Broadcast scalar to match tensor precision and perform fused computation
    result = in_1 * in_0_val + in_3
    
    # Store result - ensure correct memory layout
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_op(tmp_1, tmp_0, in_3):
    # Determine the shape for processing
    # Handle both scalar and tensor case for tmp_0
    if tmp_0.numel() == 1:
        in_0_val = tmp_0.item()
    else:
        raise ValueError("tmp_0 should be a scalar parameter")
    
    # Use the shape of the tensor inputs (all should be same shape)
    tensor_shape = tmp_1.shape
    num_elements = tmp_1.numel()
    
    # Adaptive block size based on tensor size
    if num_elements < 1_000_000:
        BLOCK_SIZE = 1024
    elif num_elements < 10_000_000:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096  # For very large tensors
    
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(tmp_1)
    
    # Launch kernel - tmp_1 already contains (in_2 + in_1) result
    fused_kernel[(num_programs,)](
        in_1_ptr=tmp_1,  # tmp_1 contains (in_2 + in_1) result
        in_3_ptr=in_3,    # the third input tensor
        out_ptr=out,
        in_0_val=in_0_val,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_op