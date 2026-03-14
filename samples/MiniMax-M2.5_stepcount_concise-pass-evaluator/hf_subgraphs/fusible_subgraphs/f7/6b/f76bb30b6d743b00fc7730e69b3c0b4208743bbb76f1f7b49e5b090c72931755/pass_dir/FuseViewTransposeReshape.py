import torch
import triton
import triton.language as tl

# Pattern matching function - matches the view-transpose-reshape pattern
# This pattern is used in attention for Q/K/V projection reshape
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the computation pattern:
    1. Linear transformation on input
    2. View + Transpose + Reshape on key_states
    3. View + Transpose + Reshape on linear output (value)
    4. View + Transpose + Reshape on query_states
    5. Final transpose on key
    
    The pattern returns (query, key, value) in the attention format.
    """
    # Linear layer
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_1 = tmp_0 = None
    
    # Key states: view -> transpose -> reshape -> transpose
    tmp_3 = in_3.view(1, -1, 16, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_3 = None
    
    # Value (from linear): view -> transpose -> reshape
    tmp_5 = tmp_2.view(1, -1, 16, 64)
    tmp_2 = None
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_5 = None
    
    # Query states: view -> transpose -> reshape
    tmp_7 = in_4.view(1, 1, 16, 64)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_7 = None
    tmp_9 = tmp_8.reshape(16, -1, 64)
    tmp_8 = None
    
    # Final reshape and transpose for key
    tmp_10 = tmp_4.reshape(16, -1, 64)
    tmp_4 = None
    tmp_11 = tmp_6.reshape(16, -1, 64)
    tmp_6 = None
    tmp_12 = tmp_10.transpose(1, 2)
    tmp_10 = None
    
    return (tmp_9, tmp_12, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments needed for the replacement function."""
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_reshape_transpose_kernel(
    input_ptr, output_ptr, 
    seq_len: tl.constexpr, 
    heads: tl.constexpr, 
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs view + transpose + reshape in a single operation.
    
    Input shape: [batch, seq, heads * head_dim] = [1, seq, 1024]
    Output shape: [heads, seq, head_dim] = [16, seq, 64]
    
    The operation is equivalent to:
    input.view(1, seq, heads, head_dim).transpose(1, 2).reshape(heads, seq, head_dim)
    """
    # Each program handles one head
    head_id = tl.program_id(0)
    
    # Calculate output strides
    # Output shape: [heads, seq, head_dim]
    # Stride for head dimension: seq * head_dim
    # Stride for seq dimension: head_dim
    
    for seq_idx in range(seq_len):
        # Calculate offsets
        # Input: [batch, seq, heads * head_dim] -> linear offset = seq_idx * (heads * head_dim) + head_id * head_dim + dim
        input_base = seq_idx * (heads * head_dim) + head_id * head_dim
        
        # Output: [heads, seq, head_dim] -> offset = head_id * (seq * head_dim) + seq_idx * head_dim + dim
        output_base = head_id * (seq_len * head_dim) + seq_idx * head_dim
        
        # Process head_dim elements
        for dim_idx in range(0, head_dim, BLOCK_SIZE):
            offsets = dim_idx + tl.arange(0, BLOCK_SIZE)
            mask = offsets < head_dim
            
            # Load from input [seq, heads, head_dim] -> need to transpose
            # Input layout: [seq, heads, head_dim] but stored as [seq, heads * head_dim]
            # We want: [heads, seq, head_dim]
            load_offsets = input_base + offsets
            x = tl.load(input_ptr + load_offsets, mask=mask, other=0.0)
            
            # Store to output
            tl.store(output_ptr + output_base + offsets, x, mask=mask)


@triton.jit
def fused_reshape_transpose_kernel_key(
    input_ptr, output_ptr, 
    seq_len: tl.constexpr, 
    heads: tl.constexpr, 
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for key: view + transpose + reshape + final transpose.
    
    Input shape: [batch, seq, heads * head_dim] = [1, seq, 1024]
    Output shape: [seq, head_dim, heads] = [1, 64, 16]
    
    The operation is equivalent to:
    input.view(1, seq, heads, head_dim).transpose(1, 2).reshape(heads, seq, head_dim).transpose(1, 2)
    """
    # Each program handles one element in the head_dim * seq dimension
    pid = tl.program_id(0)
    num_elements = seq_len * head_dim
    
    head_id = pid % heads
    remainder = pid // heads
    seq_idx = remainder % seq_len
    dim_idx = remainder // seq_len
    
    # Calculate offsets
    # Input: [batch, seq, heads * head_dim]
    input_offset = seq_idx * (heads * head_dim) + head_id * head_dim + dim_idx
    
    # Output: [seq, head_dim, heads]
    # Layout: [seq_idx, dim_idx, head_id]
    output_offset = seq_idx * (head_dim * heads) + dim_idx * heads + head_id
    
    # Load and store single element
    x = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, x)


@torch.fx.wrap
def fused_reshape_transpose_wrapper(input_tensor, seq_len, heads, head_dim, output_tensor):
    """
    Wrapper function to launch the fused reshape-transpose kernel.
    """
    # Use one program per head
    grid = (heads,)
    
    BLOCK_SIZE = 64
    
    fused_reshape_transpose_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )


@torch.fx.wrap
def fused_reshape_transpose_key_wrapper(input_tensor, seq_len, heads, head_dim, output_tensor):
    """
    Wrapper function to launch the fused kernel for key (with final transpose).
    """
    # Use enough programs to cover all elements
    num_elements = seq_len * head_dim * heads
    grid = (num_elements,)
    
    BLOCK_SIZE = 1
    
    fused_reshape_transpose_kernel_key[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )


@torch.fx.wrap
def linear_and_reshape_kernel_wrapper(bias, weight, hidden_states, key_states, query_states):
    """
    Combined function that:
    1. Applies linear transformation to hidden_states
    2. Performs fused reshape + transpose for all three tensors (key, value, query)
    3. Applies final transpose to key
    
    Returns (query, key, value) in attention format.
    """
    # Get dimensions
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_dim = hidden_states.shape[2]
    heads = 16
    head_dim = 64
    
    # Ensure seq_len is correct for each input
    key_seq_len = key_states.shape[1]
    query_seq_len = query_states.shape[1]
    
    # Apply linear transformation
    # PyTorch linear: y = xW^T + b
    # hidden_states: [1, seq, 1024], weight: [1024, 1024] -> result: [1, seq, 1024]
    value = torch.nn.functional.linear(hidden_states, weight, bias)
    
    # Allocate output tensors
    # Query: [heads, query_seq, head_dim] = [16, 1, 64]
    query_output = torch.empty([heads, query_seq_len, head_dim], device=query_states.device, dtype=query_states.dtype)
    # Key: [key_seq, head_dim, heads] = [1, 64, 16] (after final transpose)
    key_output = torch.empty([key_seq_len, head_dim, heads], device=key_states.device, dtype=key_states.dtype)
    # Value: [heads, seq, head_dim] = [16, 1, 64]
    value_output = torch.empty([heads, seq_len, head_dim], device=value.device, dtype=value.dtype)
    
    # Process query
    fused_reshape_transpose_wrapper(query_states, query_seq_len, heads, head_dim, query_output)
    
    # Process key
    fused_reshape_transpose_key_wrapper(key_states, key_seq_len, heads, head_dim, key_output)
    
    # Process value
    fused_reshape_transpose_wrapper(value, seq_len, heads, head_dim, value_output)
    
    return (query_output, key_output, value_output)


def replacement_func():
    """Returns the replacement function."""
    return linear_and_reshape_kernel_wrapper