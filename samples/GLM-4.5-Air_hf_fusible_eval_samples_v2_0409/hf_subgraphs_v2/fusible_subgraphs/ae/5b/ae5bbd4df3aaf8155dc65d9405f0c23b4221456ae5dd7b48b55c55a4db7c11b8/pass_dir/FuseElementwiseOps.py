import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_2):
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14

def replacement_args(tmp_5, in_2):
    return (tmp_5, in_2)

@triton.jit
def fused_elementwise_kernel(
    input_ptr,      # [1, num_heads, seq_len, 2] - sigmoid input
    const_ptr,      # [1, num_heads, 1, 1] - constant multiplier
    output_ptr,     # [1, num_heads, seq_len * 2, 1] - final output
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
):
    pid_k = tl.program_id(0)  # head index
    pid_m = tl.program_id(1)  # sequence position (each position handles 2 output elements)
    
    # Load input values for this head at this position
    # Get both elements (we have 2 elements per position to process)
    input_offset = (pid_k * seq_len * 2 + pid_m * 2)
    
    # Load the two input elements
    input_val_0 = tl.load(input_ptr + input_offset)
    input_val_1 = tl.load(input_ptr + input_offset + 1)
    
    # Load constant multiplier (broadcasted)
    const_offset = pid_k  # [1, num_heads, 1, 1] -> just head index
    const_val = tl.load(const_ptr + const_offset)
    
    # Apply sigmoid to both elements
    sigmoid_val_0 = 1.0 / (1.0 + tl.exp(-input_val_0))
    sigmoid_val_1 = 1.0 / (1.0 + tl.exp(-input_val_1))
    
    # Split into chunks (first element is chunk0, second is chunk1)
    chunk0 = sigmoid_val_0
    chunk1 = sigmoid_val_1
    
    # Apply element-wise operations
    # chunk1 * const_val - 1.0
    intermediate_val = chunk1 * const_val - 1.0
    
    # chunk0 * intermediate_val + 2.0
    final_val_0 = chunk0 * intermediate_val + 2.0
    final_val_1 = chunk1 * const_val - 1.0 + 2.0  # For the second element, apply same operations
    
    # Store results - reshape to [1, num_heads, seq_len * 2, 1]
    output_offset = (pid_k * seq_len * 2 + pid_m * 2 + pid_m)
    
    # Store first result
    result_offset_0 = pid_k * seq_len * 2 + pid_m * 2
    tl.store(output_ptr + result_offset_0, final_val_0)
    
    # Store second result  
    result_offset_1 = pid_k * seq_len * 2 + pid_m * 2 + 1
    tl.store(output_ptr + result_offset_1, final_val_1)

def kernel_wrapper(input_vals, const_vals):
    # Get tensor shapes
    num_heads = input_vals.shape[1]
    seq_len = input_vals.shape[2]
    
    # Output shape: [1, num_heads, seq_len * 2, 1]
    output_shape = [1, num_heads, seq_len * 2, 1]
    output = torch.empty(output_shape, dtype=input_vals.dtype, device=input_vals.device)
    
    # Flatten tensors for easier indexing (remove batch dimension)
    # input_vals: [1, num_heads, seq_len, 2] -> [num_heads, seq_len, 2]
    input_flat = input_vals.view(num_heads, seq_len, 2)
    # const_vals: [1, num_heads, 1, 1] -> [num_heads] 
    const_flat = const_vals.view(num_heads, 1, 1)
    
    # Reshape output to [num_heads, seq_len * 2] for kernel
    output_flat = output.view(num_heads, seq_len * 2)
    
    # Grid: (head_index, sequence_position_groups)
    # Each program handles 2 input elements (corresponding to original sequence positions)
    grid = (num_heads, seq_len)
    
    # Launch kernel
    fused_elementwise_kernel[grid](
        input_flat,
        const_flat,
        output_flat,
        num_heads,
        seq_len
    )
    
    return output

@torch.fx.wrap
def fused_elementwise_operations(input_vals, const_vals):
    return kernel_wrapper(input_vals, const_vals)

def replacement_func():
    return fused_elementwise_operations