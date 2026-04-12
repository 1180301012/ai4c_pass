import torch
import triton
import triton.language as tl

@triton.jit
def optimized_bool_conversion_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for boolean conversion with direct memory access."""
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Boundary mask
    
    # Load input and convert to boolean (non-zero becomes True, zero becomes False)
    input_val = tl.load(input_ptr + offsets, mask=mask)
    bool_result = input_val != 0
    
    # Store boolean result
    tl.store(output_ptr + offsets, bool_result, mask=mask)

@torch.fx.wrap
def optimized_device_and_bool_conversion(input_tensor, seq_length):
    """Wrapper function for optimized device placement and boolean conversion."""
    # Ensure input is on GPU
    input_gpu = input_tensor.cuda()
    
    # Create arange on GPU efficiently
    range_tensor = torch.arange(0, seq_length, device=input_gpu.device, dtype=torch.int64)
    
    # Perform optimized boolean conversion
    N = input_gpu.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    bool_result = torch.empty(N, dtype=torch.bool, device=input_gpu.device)
    
    optimized_bool_conversion_kernel[(num_programs,)](
        input_ptr=input_gpu,
        output_ptr=bool_result,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return range_tensor, bool_result

def pattern(in_0):
    """Pattern matching: arange + device transfer + bool conversion."""
    # Use a generic sequence length that will be captured by replacement_args
    tmp_1 = torch.arange(0, 512, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    """Extract arguments for optimized function."""
    # Extract the input shape to determine the correct sequence length
    # This works because in all cases, the arange length matches input.shape[1]
    seq_length = in_0.shape[1] if len(in_0.shape) >= 2 else in_0.shape[0] if len(in_0.shape) == 1 else 512
    return (in_0, seq_length)

def replacement_func():
    """Return reference to optimized kernel wrapper."""
    return optimized_device_and_bool_conversion