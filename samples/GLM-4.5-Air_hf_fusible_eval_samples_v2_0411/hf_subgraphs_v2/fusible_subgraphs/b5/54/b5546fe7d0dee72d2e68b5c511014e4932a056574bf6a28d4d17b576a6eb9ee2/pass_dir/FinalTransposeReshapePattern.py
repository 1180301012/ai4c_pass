import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern: Addition followed by transpose(1,2) and reshape"""
    tmp_9 = input_tensor
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 152)
    return tmp_11

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def final_transpose_reshape_kernel(
    input_ptr, output_ptr,
    input_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel that combines transpose(1,2) and reshape operations
    Input: (1, 197, 152) -> transpose(1,2) -> (1, 152, 197) -> reshape -> (1, 197*152)
    """
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load input data
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply transpose(1,2) and reshape logic
    # Input shape: (1, 197, 152) -> transpose(1,2) -> (1, 152, 197) -> reshape -> (1, 197*152)
    
    # For transpose(1,2) on (1, 197, 152):
    # Original indices: [batch, channel1, channel2]
    # Transposed indices: [batch, channel2, channel1]
    
    # Calculate indices for shape (1, 197, 152)
    orig_index = offsets % (197 * 152)  # Ignore batch (always 1)
    
    # Original: channel1 = orig_index // 152, channel2 = orig_index % 152
    channel1 = orig_index // 152
    channel2 = orig_index % 152
    
    # After transpose(1,2): (1, 152, 197)
    # New index = channel2 * 197 + channel1
    transposed_index = channel2 * 197 + channel1
    
    # After reshape(1, -1): (1, 197*152) = (1, 29944)
    # The reshape operation is straightforward since it's just flattening
    reshaped_index = transposed_index
    
    # Store results
    tl.store(output_ptr + reshaped_index, input_values, mask=mask)

@torch.fx.wrap
def final_transpose_reshape_operations(input_tensor):
    """
    Combine transpose(1,2) and reshape(1, 197, 152) operations
    """
    # Input is assumed to be (1, 197, 152) after previous operations
    input_shape = input_tensor.shape
    expected_input_shape = (1, 197, 152)
    
    # Create output tensor after all operations
    output_shape = (1, 197 * 152)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Ensure input is contiguous for efficient processing
    input_tensor_contiguous = input_tensor.contiguous()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_elements = 1 * 197 * 152  # Total elements in input
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    final_transpose_reshape_kernel[(num_programs,)](
        input_tensor_contiguous,
        output,
        num_elements,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return final_transpose_reshape_operations