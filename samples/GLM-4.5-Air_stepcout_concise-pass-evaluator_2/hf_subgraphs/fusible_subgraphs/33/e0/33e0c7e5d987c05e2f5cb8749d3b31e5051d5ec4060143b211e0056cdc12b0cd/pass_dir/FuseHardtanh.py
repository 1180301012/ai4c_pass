import torch
import triton
import triton.language as tl

# Pattern matching function - matches hardtanh operation
def pattern(in_3):
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Optimized HardTanh kernel using Triton
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # HardTanh: clamp values between min_val and max_val
    # Using tl.minimum and tl.maximum for efficiency
    x = tl.minimum(tl.maximum(x, min_val), max_val)
    
    # Store output
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def triton_hardtanh(x, min_val=0.0, max_val=6.0):
    """Triton-accelerated HardTanh activation."""
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Flatten the tensor for 1D processing
    x_flat = x.view(-1)
    output_flat = output.view(-1)
    
    # Use 2048 blocks for good balance of parallelism and overhead
    BLOCK_SIZE = 2048
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    hardtanh_kernel[(num_programs,)](
        x_flat,
        output_flat,
        n_elements,
        min_val,
        max_val,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return triton_hardtanh