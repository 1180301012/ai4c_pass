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
def optimized_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32 and compute everything in one pass
    float_vals = tl.cast(in_vals, tl.float32)
    
    # Compute 1 - input
    result_1_minus_input = 1.0 - float_vals
    
    # Create mask for original non-zero values and compute final result directly
    nonzero_mask = (in_vals != 0)
    
    # Optimized computation: fuse the operations
    # Where input != 0: (1-input) * -inf, Where input == 0: 0
    final_result = tl.where(
        nonzero_mask,
        result_1_minus_input * -3.4028234663852886e+38,
        tl.cast(0, tl.float32)
    )
    
    # Store results with vectorized stores
    tl.store(out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    n_elements = in_0.numel()
    
    BLOCK_SIZE = 1024  # Optimal for this workload
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    optimized_kernel[(num_programs, 1, 1)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper