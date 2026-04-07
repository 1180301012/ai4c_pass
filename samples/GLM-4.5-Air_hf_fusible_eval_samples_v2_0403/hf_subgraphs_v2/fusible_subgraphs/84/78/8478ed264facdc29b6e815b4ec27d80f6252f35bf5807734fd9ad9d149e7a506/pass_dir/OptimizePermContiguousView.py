import torch
import triton
import triton.language as tl

def pattern(input_tensor, batch_size, final_shape):
    # Match the sequence: permute(0, 2, 1, 3) -> contiguous() -> view(final_shape)  
    permuted = input_tensor.permute(0, 2, 1, 3)
    contiguous_tensor = permuted.contiguous()
    result = contiguous_tensor.view(final_shape)
    return result

def replacement_args(input_tensor, batch_size, final_shape):
    return (input_tensor, batch_size, final_shape)

@triton.jit
def optimized_reshape_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len_1, seq_len_2, head_dim,
    final_dim_1, final_dim_2, final_dim_3,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (final_dim_1 * final_dim_2 * final_dim_3)
    
    # Calculate indices from flattened offset
    idx_0 = offsets // (final_dim_2 * final_dim_3)
    idx_remainder = offsets % (final_dim_2 * final_dim_3)
    idx_1 = idx_remainder // final_dim_3
    idx_2 = idx_remainder % final_dim_3
    
    # Convert to original tensor indices accounting for permute
    # Original: [batch, head_dim, seq_len_1, seq_len_2] 
    # Input: [batch, seq_len_1, seq_len_2, head_dim]
    input_idx_0 = idx_0
    input_idx_1 = idx_2  # seq_len_1 -> head_dim position
    input_idx_2 = idx_1  # head_dim -> seq_len_1 position  
    input_idx_3 = idx_2  # seq_len_2 -> head_dim stays
    
    # Calculate linear input offset
    input_offset = (input_idx_0 * seq_len_1 * seq_len_2 * head_dim + 
                   input_idx_1 * seq_len_2 * head_dim + 
                   input_idx_2 * head_dim + 
                   input_idx_3)
    
    # Load data with permutation logic built-in
    if input_idx_0 < batch_size and input_idx_1 < seq_len_1 and input_idx_2 < seq_len_2 and input_idx_3 < head_dim:
        data = tl.load(input_ptr + input_offset, mask=True)
    else:
        data = tl.full([1], 0.0, dtype=tl.float32)
    
    # Store result
    output_offset = offsets
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap  
def optimized_reshape(input_tensor, batch_size, final_shape):
    # Get input tensor dimensions assuming it's the output from attention
    # Input shape: [batch, num_heads, seq_len, head_dim]
    batch, num_heads, seq_len, head_dim = input_tensor.shape
    
    # Calculate final shape
    final_dim_1, final_dim_2, final_dim_3 = final_shape
    
    # Create output tensor
    output = torch.empty(final_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate block size and number of programs
    total_elements = final_dim_1 * final_dim_2 * final_dim_3
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch,
        seq_len_1=num_heads,      # Position after permute becomes seq_len_1
        seq_len_2=seq_len,       # Original seq_len becomes seq_len_2  
        head_dim=head_dim,       # head_dim stays head_dim
        final_dim_1=final_dim_1,
        final_dim_2=final_dim_2,
        final_dim_3=final_dim_3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_reshape