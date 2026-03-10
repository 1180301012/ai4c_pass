import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the entire computation graph
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    tmp_3 = tmp_2.t()
    return (tmp_2, tmp_3)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_kernel(
    in_0_ptr,
    in_1_ptr, 
    in_2_ptr,
    out_ptr_2,  # tmp_2 output: [2,1]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    # in_0 and in_1 are scalars, in_2 is [2,1] flattened to [2]
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Load scalars (they'll be broadcasted)
    in_0 = tl.load(in_0_ptr)
    in_1 = tl.load(in_1_ptr)
    
    # fused computation: exp(in_1) * in_2 + in_0
    tmp_0 = tl.exp(in_1)  # exp(scalar)
    tmp_1 = in_2 * tmp_0  # [2] * scalar -> [2]
    tmp_2 = tmp_1 + in_0  # [2] + scalar -> [2]
    
    # Store tmp_2 result (original [2,1] flattened)
    tl.store(out_ptr_2 + offsets, tmp_2, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    # Handle the [2,1] tensor - flatten for processing
    in_2_flat = in_2.flatten()
    n_elements = in_2_flat.numel()
    
    # Create output tensor for tmp_2 (the fused result)
    tmp_2_out = torch.empty_like(in_2)  # [2,1]
    
    # Set up launch configuration
    BLOCK_SIZE = 32  # Optimal for small tensors
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel to compute fused operations
    fused_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2_flat,
        out_ptr_2=tmp_2_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Perform transpose as a separate operation (cheap for small tensors)
    tmp_3_out = tmp_2_out.t()  # [2,1] -> [1,2]
    
    return tmp_2_out, tmp_3_out

def replacement_func():
    return kernel_wrapper