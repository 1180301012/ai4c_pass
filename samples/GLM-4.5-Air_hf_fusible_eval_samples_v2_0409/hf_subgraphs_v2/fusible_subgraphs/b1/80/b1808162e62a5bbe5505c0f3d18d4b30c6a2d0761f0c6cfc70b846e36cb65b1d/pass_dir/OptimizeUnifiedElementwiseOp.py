import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Exact mirror of the computation in model.py
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return (tmp_4,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute offsets within the block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32
    x_float = x.to(tl.float32)
    
    # Compute the unified operation:
    # tmp_1 = 1 - x_float
    # tmp_2 = tmp_1.bool() (True when tmp_1 != 0, False when tmp_1 == 0)
    # tmp_3 = tmp_1.masked_fill(tmp_2, -inf) -> this fills where tmp_2 is True with -inf
    # tmp_4 = tmp_3 * tmp_1 -> this means: where tmp_2 is False, tmp_4 = tmp_1 * tmp_1, where tmp_2 is True, tmp_4 = -inf * tmp_1
    tmp_1 = 1.0 - x_float
    # When tmp_1 == 0: result = 0 * 0 = 0
    # When tmp_1 != 0: result = -inf * tmp_1 = -inf  
    result = tl.where(tmp_1 == 0, tmp_1 * tmp_1, -3.4028234663852886e+38 * tmp_1)
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    # Get input tensor metadata
    n_elements = in_0.numel()
    
    # Further optimize block size threshold for better GPU utilization
    BLOCK_SIZE = 512 if n_elements <= 1024 else 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch the optimized kernel
    optimized_kernel[(num_programs,)](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return kernel_wrapper