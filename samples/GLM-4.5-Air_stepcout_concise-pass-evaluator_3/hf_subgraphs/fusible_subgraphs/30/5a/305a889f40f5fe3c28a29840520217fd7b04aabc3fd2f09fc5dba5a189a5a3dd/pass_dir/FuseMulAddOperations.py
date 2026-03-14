import torch
import triton
import triton.language as tl

def pattern(reshape_input, reshape_shape, mul_input, add_input):
    """
    Pattern that matches: reshape -> multiply -> add
    
    This pattern appears in all graphs:
    tmp_4 = reshape_input.reshape(reshape_shape)
    tmp_5 = tmp_4 * mul_input
    tmp_6 = add_input + tmp_5
    """
    # Original operations
    tmp_4 = reshape_input.reshape(reshape_shape)
    tmp_5 = tmp_4 * mul_input
    tmp_6 = add_input + tmp_5
    return tmp_6

@triton.jit
def fused_mul_add_kernel(
    reshape_ptr,
    mul_ptr,
    add_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines reshape, multiplication, and addition
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Get shapes from the tensor pointers
    reshape_shape = tl.device_shape(reshape_ptr)
    mul_shape = tl.device_shape(mul_ptr)
    add_shape = tl.device_shape(add_ptr)
    out_shape = tl.device_shape(out_ptr)
    
    # Calculate total elements and work per program
    total_elements = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate start and end indices for this program
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, total_elements)
    
    if start_idx >= total_elements:
        return
    
    # Helper function to get multi-dimensional indices
    def get_indices(idx, shape):
        # Unflatten 1D index to 4D indices
        total = idx
        f = total // (shape[1] * shape[2] * shape[3])
        total = total % (shape[1] * shape[2] * shape[3])
        h = total // (shape[2] * shape[3])
        total = total % (shape[2] * shape[3])
        w = total // shape[3]
        c = total % shape[3]
        return f, h, w, c
    
    # Process elements in this program
    for idx in range(start_idx, end_idx):
        f, h, w, c = get_indices(idx, out_shape)
        
        # For reshape input - assume it's already in the correct output shape
        reshape_offset = idx
        reshape_val = tl.load(reshape_ptr + reshape_offset, mask=(f < reshape_shape[0] and h < reshape_shape[1] and w < reshape_shape[2] and c < reshape_shape[3]), other=0.0)
        
        # For mul input ( broadcasting from [seq_len, features] to [batch, heads, seq_len, features])
        mul_w, mul_c = w, c  # Same sequence and feature indices
        mul_offset = mul_w * mul_shape[1] + mul_c
        mul_val = tl.load(mul_ptr + mul_offset, mask=(mul_w < mul_shape[0] and mul_c < mul_shape[1]), other=0.0)
        
        # Multiply
        mul_result = reshape_val * mul_val
        
        # For add input (same shape as output)
        add_offset = f * add_shape[1] * add_shape[2] * add_shape[3] + \
                     h * add_shape[2] * add_shape[3] + \
                     w * add_shape[3] + c
        add_val = tl.load(add_ptr + add_offset, mask=(f < add_shape[0] and h < add_shape[1] and w < add_shape[2] and c < add_shape[3]), other=0.0)
        
        # Add
        result = add_val + mul_result
        
        # Store result
        tl.store(out_ptr + idx, result)

@torch.fx.wrap
def fused_mul_add(reshape_input, reshape_shape, mul_input, add_input):
    """Wrapper for the fused multiply-add kernel"""
    # Validate input shapes
    if len(reshape_input.shape) != 4 or len(add_input.shape) != 4:
        # Fall back to original computation if shapes are unexpected
        tmp_4 = reshape_input.reshape(reshape_shape)
        tmp_5 = tmp_4 * mul_input
        tmp_6 = add_input + tmp_5
        return tmp_6
    
    # Get shapes
    reshape_shape = reshape_input.shape  # Note: reshape_input is already reshaped
    mul_shape = mul_input.shape
    add_shape = add_input.shape
    
    # Output shape is same as add_shape
    out_shape = add_shape
    
    # Create output tensor
    output = torch.empty(out_shape, dtype=torch.float32, device=reshape_input.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    total_elements = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_mul_add_kernel[(num_programs,)](
        reshape_input,
        mul_input,
        add_input,
        output,
        BLOCK_SIZE,
    )
    
    return output

def replacement_args(reshape_input, reshape_shape, mul_input, add_input):
    return (reshape_input, reshape_shape, mul_input, add_input)

def replacement_func():
    return fused_mul_add