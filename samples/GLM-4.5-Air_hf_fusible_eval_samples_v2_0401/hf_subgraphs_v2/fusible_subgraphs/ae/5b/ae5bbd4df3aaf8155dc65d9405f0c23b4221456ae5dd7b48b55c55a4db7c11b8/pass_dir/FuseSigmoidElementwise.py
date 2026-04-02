import torch
import triton
import triton.language as tl

# Pattern to match: sigmoid followed by chunking and element-wise operations
def pattern(tmp_5_input, in_2_input):
    tmp_6 = torch.sigmoid(tmp_5_input)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2_input
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    return tmp_13

# Extract arguments for the replacement function
def replacement_args(tmp_5_input, in_2_input):
    return (tmp_5_input, in_2_input)

# Optimized Triton kernel for fused sigmoid+elementwise operations
@triton.jit
def fused_sigmoid_elementwise_kernel(
    sigmoid_input_ptr,  # Input [1, n_heads, seq_len, 2] to sigmoid
    in_2_ptr,           # Input [1, n_heads, 1, 1] for scaling
    output_ptr,         # Output [1, n_heads, seq_len, 1] after all operations
    n_heads,            # Number of heads
    seq_len,            # Sequence length (199)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element at a time
    pid = tl.program_id(0)
    
    # Total elements in output: [1, n_heads, seq_len, 1]
    total_elements = n_heads * seq_len
    mask = pid < total_elements
    
    if not mask:
        return
    
    # Decode output index into head, sequence
    head_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Load the two elements from the chunk (dim=0 and dim=1)
    base_offset = head_idx * seq_len * 2 + seq_idx * 2
    
    # Load both elements needed for the computation
    sigmoid_first = tl.load(sigmoid_input_ptr + base_offset + 0, mask=True, other=0.0)
    sigmoid_second = tl.load(sigmoid_input_ptr + base_offset + 1, mask=True, other=0.0)
    
    # Apply sigmoid (convert to fp32 for precision)
    sigmoid_first_fp32 = sigmoid_first.to(tl.float32)
    sigmoid_first_fp32 = tl.sigmoid(sigmoid_first_fp32)
    sigmoid_first = sigmoid_first_fp32.to(sigmoid_first.dtype)
    
    sigmoid_second_fp32 = sigmoid_second.to(tl.float32)
    sigmoid_second_fp32 = tl.sigmoid(sigmoid_second_fp32)
    sigmoid_second = sigmoid_second_fp32.to(sigmoid_second.dtype)
    
    # Load in_2 value for this head
    in_2_val = tl.load(in_2_ptr + head_idx, mask=True, other=0.0)
    
    # Apply element-wise operations:
    # tmp_last = sigmoid_second * in_2_val
    # tmp_last = tmp_last - 1.0
    # result = sigmoid_first * tmp_last
    # result = result + 2.0
    tmp_last = sigmoid_second * in_2_val
    tmp_last = tmp_last - 1.0
    result = sigmoid_first * tmp_last
    result = result + 2.0
    
    # Store result - only the first half (chunk[0] results)
    # Output shape: [1, n_heads, seq_len, 1]
    tl.store(output_ptr + pid, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_elementwise_triton(tmp_5_input, in_2_input):
    # Get tensor properties - generic for different head counts
    n_heads = tmp_5_input.shape[1]  # Can be 12 or 16
    seq_len = tmp_5_input.shape[2]   # 199
    
    # Output shape: [1, n_heads, seq_len, 1] (tmp_13 after element-wise operations)
    output_shape = (1, n_heads, seq_len, 1)
    output = torch.empty(output_shape, dtype=tmp_5_input.dtype, device=tmp_5_input.device)
    
    # Calculate grid size - each program handles one output element
    total_elements = n_heads * seq_len
    
    # Launch kernel - one program per output element
    fused_sigmoid_elementwise_kernel[(total_elements,)](
        sigmoid_input_ptr=tmp_5_input,
        in_2_ptr=in_2_input,
        output_ptr=output,
        n_heads=n_heads,
        seq_len=seq_len,
        BLOCK_SIZE=1,  # Not used in kernel, but required by syntax
    )
    
    return output

# Replacement function that returns the optimized kernel
def replacement_func():
    return fused_sigmoid_elementwise_triton