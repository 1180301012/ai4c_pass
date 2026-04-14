import torch
import triton
import triton.language as tl

def pattern(tmp_9, in_1, tmp_2):
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13

def replacement_args(tmp_9, in_1, tmp_2):
    return (tmp_9, in_1, tmp_2)

@triton.jit
def scaling_kernel(
    normalized_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate global indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load normalized values and bias
    normalized = tl.load(normalized_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr)
    
    # Add bias and apply scaling
    scaled = normalized * (1.0 + bias)
    
    # Store result (will handle dtype conversion in wrapper)
    tl.store(output_ptr + offsets, scaled, mask=mask)

@torch.fx.wrap
def fused_scaling(tmp_9, in_1, tmp_2):
    # Calculate total elements
    n_elements = tmp_9.numel()
    
    # Use larger block size for better GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor in float32 (kernel works with float32)
    output_float = torch.empty(tmp_9.shape, dtype=torch.float32, device=tmp_9.device)
    
    # Launch kernel
    scaling_kernel[(num_programs,)](
        normalized_ptr=tmp_9,
        bias_ptr=in_1,
        output_ptr=output_float,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Convert to target dtype (same as tmp_2)
    output = output_float.type_as(tmp_2)
    
    return output

def replacement_func():
    return fused_scaling