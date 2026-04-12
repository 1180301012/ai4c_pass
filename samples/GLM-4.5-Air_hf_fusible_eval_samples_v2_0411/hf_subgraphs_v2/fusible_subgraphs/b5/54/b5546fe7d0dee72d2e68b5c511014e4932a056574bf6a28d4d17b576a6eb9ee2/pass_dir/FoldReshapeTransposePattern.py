import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern: Reshape followed by transpose of last two dimensions"""
    reshaped = input_tensor.reshape(1, 8, 19, 196)
    result = reshaped.transpose(-1, -2)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_reshape_transpose_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Kernel that combines reshape and transpose operations"""
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply the reshape and transpose logic
    # For shape (1, 8, 19, 196) -> transpose last two dims to (1, 8, 196, 19)
    # We need to compute the correct mapping of indices
    input_size = 8 * 19 * 196  # Total elements per batch
    batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
    
    # Calculate the output indices after reshape + transpose
    # Assuming input is flattened in the right order for reshape(1, 8, 19, 196)
    orig_index = offsets % input_size
    
    # Reshape indices: (1, 8, 19, 196)
    channel = orig_index // (19 * 196)  # 0-7
    height = (orig_index % (19 * 196)) // 196  # 0-18
    width = orig_index % 196  # 0-195
    
    # Transpose last two dimensions: swap height and width
    new_index = channel * (196 * 19) + width * 19 + height
    
    # Add batch offset if needed
    batch_offset = (offsets // input_size) * (8 * 196 * 19)
    final_index = batch_offset + new_index
    
    # Store results
    tl.store(output_ptr + final_index, input_values, mask=mask)

@torch.fx.wrap
def fused_reshape_transpose(input_tensor):
    """
    Combines reshape(1, 8, 19, 196) and transpose(-1, -2) operations
    """
    # Get input tensor size
    input_size = input_tensor.numel()
    
    # Calculate output shape after operations: (1, 8, 196, 19)
    output_shape = (1, 8, 196, 19)
    output_size = 1 * 8 * 196 * 19
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Ensure input is contiguous for efficient processing
    input_tensor_contiguous = input_tensor.contiguous()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_transpose_kernel[(num_programs,)](
        input_tensor_contiguous,
        output,
        input_size,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_reshape_transpose