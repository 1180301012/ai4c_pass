import torch
import triton
import triton.language as tl


def pattern(input_tensor, batch_size, source_tensor):
    # Reshape operation
    reshaped = input_tensor.reshape(batch_size, 16, 16, -1)
    # Permute operation  
    permuted = reshaped.permute(0, 3, 1, 2)
    # Contiguous operation
    contiguous_result = permuted.contiguous()
    reshaped = permuted = None
    
    # Additional operations for the second output
    flattened = source_tensor.flatten(2)
    transposed = flattened.transpose(1, 2)
    flattened = None
    
    return contiguous_result, transposed


def replacement_args(input_tensor, batch_size, source_tensor):
    return (input_tensor, batch_size, source_tensor)


@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    source_ptr,
    final_out1_ptr,
    final_out2_ptr,
    batch_size: tl.constexpr,
    BLOCK_SIZE_1: tl.constexpr,
    BLOCK_SIZE_2: tl.constexpr,
):
    # Handle the first output (optimized reshape/permute/contiguous)
    pid1 = tl.program_id(0)
    
    # For first output: need to figure out original dimensions
    # Input is [batch_size, 256, 512], output is [batch_size, -1, 16, 16] then permuted to [batch_size, 16, 16, -1]
    # Final after permute(0, 3, 1, 2) should be [batch_size, -1, 16, 16]
    hidden_size = 256 * 512 // (16 * 16)  # Calculate feature dimension
    total_elements = batch_size * hidden_size * 16 * 16
    
    # Calculate position in final tensor [batch_size, hidden_size, 16, 16]
    elem_idx = pid1 * BLOCK_SIZE_1
    batch = elem_idx // (hidden_size * 16 * 16)
    remainder = elem_idx % (hidden_size * 16 * 16)
    feature = remainder // (16 * 16)
    h_2d = remainder % (16 * 16) // 16
    w_2d = remainder % 16
    
    # Calculate original position in input [batch_size, 256, 512]
    original_idx = batch * (256 * 512) + (h_2d * 16 + w_2d) * 512 + feature
    
    if original_idx < batch_size * 256 * 512:
        val = tl.load(input_ptr + original_idx, mask=None)
        # Store in final position [batch_size, hidden_size, 16, 16]
        final_pos = batch * (hidden_size * 16 * 16) + feature * (16 * 16) + h_2d * 16 + w_2d
        tl.store(final_out1_ptr + final_pos, val, mask=None)
    
    # Handle second output (flatten + transpose)
    pid2 = tl.program_id(1)
    
    # Input shape [batch_size, 64, 128, 128], output after flatten(2) and transpose(1,2)
    # flatten(2) -> [batch_size, 64, 128*128], transpose(1,2) -> [batch_size, 128*128, 64]
    batch_2 = pid2 // (128 * 64)
    remain_2 = pid2 % (128 * 64)
    height_width = remain_2 // 64  # Should be 128*128 in reality
    feature_2 = remain_2 % 64
    
    # Original position: [batch_size, 64, 128, 128]
    original_idx_2 = batch_2 * (64 * 128 * 128) + feature_2 * (128 * 128) + (height_width // 128) * 128 + (height_width % 128)
    
    if batch_2 < batch_size and feature_2 < 64 and height_width < 128 * 128:
        val_2 = tl.load(source_ptr + original_idx_2, mask=None)
        # Final position after flatten(2) + transpose(1,2): [batch_size, 128*128, 64]
        final_pos_2 = batch_2 * (128 * 64) + height_width * 64 + feature_2
        tl.store(final_out2_ptr + pid2, val_2, mask=None)


@torch.fx.wrap
def optimized_reshape_permute_contiguous(input_tensor, batch_size, source_tensor):
    # Calculate output shapes
    hidden_size = 256 * 512 // (16 * 16)  # Should be 512
    output1_size = batch_size * hidden_size * 16 * 16
    output2_size = batch_size * 64 * (128 * 128)  # After flatten, before transpose
    
    # Create output tensors
    output1 = torch.empty((batch_size, hidden_size, 16, 16), dtype=input_tensor.dtype, device=input_tensor.device)
    output2 = torch.empty((batch_size, 128*128, 64), dtype=source_tensor.dtype, device=source_tensor.device)
    
    # Configure grid parameters
    BLOCK_SIZE_1 = 512
    BLOCK_SIZE_2 = 1024
    
    num_programs_1 = (output1_size + BLOCK_SIZE_1 - 1) // BLOCK_SIZE_1
    num_programs_2 = output2_size // BLOCK_SIZE_2
    
    optimized_reshape_kernel[(num_programs_1, num_programs_2)](
        input_tensor,
        source_tensor,
        output1,
        output2,
        batch_size,
        BLOCK_SIZE_1,
        BLOCK_SIZE_2
    )
    
    return output1, output2


def replacement_func():
    return optimized_reshape_permute_contiguous