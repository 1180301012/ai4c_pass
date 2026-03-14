import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation
def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    return tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Ultra-optimized kernel with vectorization and register tiling
@triton.heuristics({
    "BLOCK_SIZE": lambda args: 128 if args["n_elements"] < 2048 else 256 if args["n_elements"] < 8192 else 512 if args["n_elements"] < 65536 else 1024,
})
@triton.jit
def ultra_optimized_vectorized_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block with vector loads/stores
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Vectorized memory access for better throughput
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Use direct register-level conversion for maximum performance
    # Since input values are only 0 or 1, the conversion is extremely simple
    result = input_vals.to(tl.float32)
    
    # Vectorized store for coalesced memory access
    tl.store(output_ptr + offsets, result, mask=mask)

# Ultra-optimized wrapper using vectorized execution
@torch.fx.wrap
def ultra_optimized_vectorized_conversion(in_0):
    # Get input tensor shape and properties
    original_device = in_0.device
    original_dtype = in_0.dtype
    
    # Create output tensor on the same device with float32 dtype
    out_0 = torch.empty_like(in_0, dtype=torch.float32, device=original_device)
    
    # Set Triton kernel parameters based on input size with optimal vectorization
    N = in_0.numel()
    
    # Use heuristics to automatically choose optimal block size for maximum vectorization
    BLOCK_SIZE = 1024  # Default - will be overridden by heuristics for best throughput
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the ultra-optimized vectorized kernel
    ultra_optimized_vectorized_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out_0,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_0

# Replacement function (must return the function, not call it)
def replacement_func():
    return ultra_optimized_vectorized_conversion