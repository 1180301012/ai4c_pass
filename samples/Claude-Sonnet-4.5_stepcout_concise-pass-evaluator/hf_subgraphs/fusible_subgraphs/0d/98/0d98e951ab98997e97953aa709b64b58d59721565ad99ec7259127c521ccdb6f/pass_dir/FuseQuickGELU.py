import torch
import triton
import triton.language as tl


# Pattern matching function - matches the QuickGELU computation pattern
# This includes: 1.702 * x -> sigmoid -> x * sigmoid -> dropout(p=0.0)
def pattern(in_0):
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3


# Extract arguments from the matched pattern
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for QuickGELU: x * sigmoid(1.702 * x)
# Using tl.sigmoid instead of torch.sigmoid to avoid API blocking
@triton.jit
def quickgelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # QuickGELU: x * sigmoid(1.702 * x)
    # Using tl.sigmoid which is allowed
    scaled_x = x * tl.constexpr(1.702)
    sig = tl.sigmoid(scaled_x)
    out = x * sig
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def quickgelu_wrapper(in_0):
    """Fused QuickGELU kernel: x * sigmoid(1.702 * x)"""
    n_elements = in_0.numel()
    
    # Use a single program to minimize kernel launch overhead
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_programs == 0:
        num_programs = 1
    
    out = torch.empty_like(in_0)
    
    quickgelu_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return quickgelu_wrapper