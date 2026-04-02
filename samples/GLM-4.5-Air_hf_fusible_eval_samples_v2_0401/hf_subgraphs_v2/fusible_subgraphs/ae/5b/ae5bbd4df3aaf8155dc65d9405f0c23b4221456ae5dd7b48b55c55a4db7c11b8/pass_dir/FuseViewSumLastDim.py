import torch
import triton
import triton.language as tl

# Pattern to match: view operation followed by sum along last dimension
def pattern(linear_tensor):
    tmp_4 = linear_tensor.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    return tmp_5

# Extract arguments for the replacement function
def replacement_args(linear_tensor):
    return (linear_tensor,)

# Optimized Triton kernel for fused view+sum operation
@triton.jit
def fused_view_sum_kernel(
    input_ptr,     # Linear transformation result [1, 12, 199, 64]
    output_ptr,    # Output [1, 12, 199, 2] after view and sum
    n_heads,       # Number of heads (12)
    seq_len,       # Sequence length (199)
    head_dim,      # Head dimension (64)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element at a time
    pid = tl.program_id(0)
    
    # Total output elements: [1, n_heads, seq_len, 2]
    total_elements = n_heads * seq_len * 2
    mask = pid < total_elements
    
    if not mask:
        return
    
    # Decode position: [head, seq, dim2]
    head_idx = pid // (seq_len * 2)
    seq_idx = (pid % (seq_len * 2)) // 2
    dim2_idx = (pid % (seq_len * 2)) % 2  # 0 or 1
    
    # In the view [1, n_heads, seq_len, 2, 4], we want to sum along the last dimension
    # For a fixed [head_idx, seq_idx, dim2_idx], we sum over 4 consecutive elements
    # In the original tensor [1, n_heads, seq_len, 64], these elements are at:
    # base = head_idx * seq_len * 64 + seq_idx * 64 + dim2_idx * 32
    base_offset = head_idx * seq_len * head_dim + seq_idx * head_dim + dim2_idx * 32
    
    # Load 4 consecutive elements that correspond to the 4 elements in the last dimension
    elem_indices = base_offset + tl.arange(0, 4)
    input_vals = tl.load(input_ptr + elem_indices, mask=tl.arange(0, 4) < 4, other=0.0)
    
    # Sum the 4 elements (preserve original dtype)
    sum_result = tl.sum(input_vals, axis=0)
    
    # Store result with original dtype
    tl.store(output_ptr + pid, sum_result, mask=mask)

@torch.fx.wrap
def fused_view_sum_triton(linear_tensor):
    # Get tensor properties
    n_heads = linear_tensor.shape[1]  # 12
    seq_len = linear_tensor.shape[2]   # 199
    head_dim = linear_tensor.shape[3]  # 64
    
    # Output shape: [1, n_heads, seq_len, 2]
    output_shape = (1, n_heads, seq_len, 2)
    output = torch.empty(output_shape, dtype=linear_tensor.dtype, device=linear_tensor.device)
    
    # Calculate grid size - each program handles one output element
    total_elements = n_heads * seq_len * 2
    
    # Launch kernel - one program per output element
    fused_view_sum_kernel[(total_elements,)](
        input_ptr=linear_tensor,
        output_ptr=output,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=1,  # Not used in kernel, but required by syntax
    )
    
    return output

# Replacement function that returns the optimized kernel
def replacement_func():
    return fused_view_sum_triton