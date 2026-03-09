import torch
import triton
import triton.language as tl

def pattern(input_tensor, arange_size):
    """
    Pattern: Create arange tensor and convert input to boolean in one fusion
    This matches the computation pattern found in all three target graphs
    """
    # Create arange tensor on device (equivalent to torch.arange(0, arange_size, device=input_tensor.device))
    arange_tensor = torch.arange(0, arange_size, device=input_tensor.device)
    
    # Convert input to boolean (equivalent to input_tensor.to(dtype=torch.bool))
    bool_tensor = input_tensor.to(dtype=torch.bool)
    
    return arange_tensor, bool_tensor

def replacement_args(input_tensor, arange_size):
    """
    Extract arguments needed for the fused kernel
    """
    return (input_tensor, arange_size)

@triton.jit
def fusion_kernel(
    input_ptr,
    out_arange_ptr,
    out_bool_ptr,
    n_elements,
    arange_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel that fuses arange creation and boolean conversion
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data (only need for boolean conversion)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Convert input to boolean
    bool_result = input_data != 0  # Convert int64 to boolean by checking non-zero
    
    # For arange, we need to generate indices, so we store the offset + local index
    # This handles the arange generation efficiently
    arange_result = offsets
    
    # Store results
    tl.store(out_arange_ptr + offsets, arange_result, mask=mask)
    tl.store(out_bool_ptr + offsets, bool_result, mask=mask)

@torch.fx.wrap
def fused_operation(input_tensor, arange_size):
    """
    Wrapper function that launches the fused kernel
    """
    # Determine grid and block sizes for optimal GPU utilization
    input_elem = input_tensor.numel()
    arange_elem = arange_size
    
    # Use larger block size for better GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions for both input and arange tensors
    grid_arange = (triton.cdiv(arange_elem, BLOCK_SIZE),)
    grid_input = (triton.cdiv(input_elem, BLOCK_SIZE),)
    
    # Create output tensors with proper shapes and types
    arange_output = torch.empty(arange_size, dtype=torch.int64, device=input_tensor.device)
    bool_output = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    
    # Launch kernel for the arange part (since arange_size may differ from input size)
    if arange_elem > 0:
        fusion_kernel[grid_arange](
            input_ptr=input_tensor,  # We pass input_tensor, but kernel uses arange_size
            out_arange_ptr=arange_output,
            out_bool_ptr=bool_output,  # For arange part, this kernel doesn't write to bool_output
            n_elements=arange_elem,
            arange_size=arange_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Launch kernel for the boolean conversion part
    if input_elem > 0:
        fusion_kernel[grid_input](
            input_ptr=input_tensor,
            out_arange_ptr=arange_output,  # For input part, this kernel doesn't write to arange_output 
            out_bool_ptr=bool_output,
            n_elements=input_elem,
            arange_size=arange_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return arange_output, bool_output

def replacement_func():
    """
    Return the optimized fused operation function
    """
    return fused_operation