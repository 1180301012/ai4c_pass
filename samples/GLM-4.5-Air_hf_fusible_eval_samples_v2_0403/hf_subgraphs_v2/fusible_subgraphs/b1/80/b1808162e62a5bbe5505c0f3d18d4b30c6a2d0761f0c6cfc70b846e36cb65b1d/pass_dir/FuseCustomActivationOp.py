import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def custom_activation_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as int64 and convert to float32 immediately
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    
    # Compute custom activation function in one kernel
    # tmp_1 = 1.0 - in_vals
    tmp_1 = 1.0 - in_vals
    
    # Create boolean mask (non-zero values) and then apply custom activation
    # Instead of separate bool conversion and masked_fill, do it directly
    non_zero_mask = tmp_1 != 0.0
    
    # Apply the activation: where non-zero, use -FLT_MAX, else original value
    activation_vals = tl.where(non_zero_mask, 
                               -3.4028234663852886e+38, 
                               tmp_1)
    
    # Final multiplication: activation_vals * tmp_1
    result = activation_vals * tmp_1
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def custom_activation_wrapper(in_0):
    # Get input tensor properties
    N = in_0.numel()
    
    # Use optimal block size for this small tensor size (484 elements)
    # For N=484, use BLOCK_SIZE=64 to minimize overhead while maintaining good occupancy
    if N <= 512:
        BLOCK_SIZE = 64  # Optimal for tensors up to 512 elements
    else:
        BLOCK_SIZE = 1024  # Standard size for larger tensors
    
    # Perfect launch configuration for small tensors
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch kernel
    custom_activation_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return custom_activation_wrapper