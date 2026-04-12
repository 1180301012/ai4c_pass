import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(sigmoid_input, in_2):
    """
    Fuse sigmoid + chunk + arithmetic operations
    """
    tmp_6 = torch.sigmoid(sigmoid_input)
    
    # Split into two parts using indexing that works with Proxy objects
    # Get the first and second elements along the last dimension
    tmp_8 = tmp_6.select(-1, 0)  # First element along last dimension
    tmp_9 = tmp_6.select(-1, 1)  # Second element along last dimension
    
    # Reshape to add the dimension back
    tmp_8 = tmp_8.unsqueeze(-1)  # Add last dimension back
    tmp_9 = tmp_9.unsqueeze(-1)
    
    # Arithmetic operations
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    
    # Final reshape for wavlm_base
    tmp_14 = tmp_13.view(1, 12, 398, 1)  # seq_len=12, final hidden=398
    return tmp_14

# Argument extraction function
def replacement_args(sigmoid_input, in_2):
    return (sigmoid_input, in_2)

# Triton kernel for fused sigmoid + arithmetic operations
@triton.jit
def fused_sigmoid_arithmetic_kernel(
    sigmoid_input_ptr,  # Input to sigmoid [batch, seq_len, hidden_size, 2]
    in_2_ptr,           # Constant tensor [batch, seq_len, 1, 1]
    output_ptr,         # Final output [batch, seq_len * 2, 1]
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total elements in final output [batch, seq_len * 2, 1]
    total_output_elements = batch * seq_len * 2
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_output_elements
    
    # Convert output offset to multi-dimensional indices
    # output_shape: [batch, seq_len * 2, 1]
    output_idx = offsets
    batch_idx = output_idx // (seq_len * 2)
    remainder = output_idx % (seq_len * 2)
    final_seq_idx = remainder // 2  # 0 or 1
    final_hidden_idx = remainder % 2
    
    # Map to sigmoid input indices [batch, seq_len, hidden_size, 2]
    sigmoid_input_batch = batch_idx
    sigmoid_input_seq = final_seq_idx  # This maps directly to sequence dim
    sigmoid_input_hidden = (final_hidden_idx + sigmoid_input_seq * hidden_size) // 2
    sigmoid_input_feature = (final_hidden_idx + sigmoid_input_seq * hidden_size) % 2
    
    # Load sigmoid input value
    sigmoid_input_offset = (sigmoid_input_batch * (seq_len * hidden_size * 2) + 
                           sigmoid_input_seq * (hidden_size * 2) + 
                           sigmoid_input_hidden * 2 + 
                           sigmoid_input_feature)
    sigmoid_val = tl.load(sigmoid_input_ptr + sigmoid_input_offset, mask=mask, other=0.0)
    
    # Apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-sigmoid_val))
    
    # Separate into two parts
    part1 = sigmoid_result
    # For part2, we need to load the other feature from the same hidden position
    part2_offset = sigmoid_input_offset + 1 if sigmoid_input_feature == 0 else sigmoid_input_offset - 1
    if sigmoid_input_feature == 0:
        # Need to load element at offset+1
        part2 = tl.load(sigmoid_input_ptr + part2_offset, mask=mask, other=0.0)
    else:
        # Need to load element at offset-1  
        part2 = tl.load(sigmoid_input_ptr + part2_offset, mask=mask, other=0.0)
    
    # Apply sigmoid to part2 as well
    part2 = 1.0 / (1.0 + tl.exp(-part2))
    
    # Load constant from in_2
    in_2_offset = (sigmoid_input_batch * (seq_len * 1 * 1) + 
                   sigmoid_input_seq * (1 * 1) + 
                   0 * 1 + 
                   0)
    const_val = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Arithmetic operations: part1 * (part2 * const_val - 1.0) + 2.0
    scaled_part2 = part2 * const_val
    adjusted_part2 = scaled_part2 - 1.0
    multiplied = part1 * adjusted_part2
    final_result = multiplied + 2.0
    
    # Store result
    tl.store(output_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_arithmetic(sigmoid_input, in_2):
    """
    Fused implementation of sigmoid + chunk + arithmetic operations
    """
    batch = 1
    seq_len = 12  # wavlm_base
    hidden_size = 199
    
    # Create output tensor [batch, seq_len * 2, 1]
    output = torch.empty(batch * seq_len * 2, dtype=sigmoid_input.dtype, device=sigmoid_input.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = batch * seq_len * 2
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sigmoid_arithmetic_kernel[(num_programs,)](
        sigmoid_input_ptr=sigmoid_input,
        in_2_ptr=in_2,
        output_ptr=output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final output [batch, seq_len, 2, 1]
    return output.view(batch, seq_len, 2, 1)

# Replacement function
def replacement_func():
    return fused_sigmoid_arithmetic