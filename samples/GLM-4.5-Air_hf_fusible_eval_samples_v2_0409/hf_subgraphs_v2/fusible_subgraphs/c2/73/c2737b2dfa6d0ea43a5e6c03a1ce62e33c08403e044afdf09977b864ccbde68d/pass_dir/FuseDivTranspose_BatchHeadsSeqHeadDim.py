import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation graph
def pattern(in_0):
    """
    Pattern for: division by scalar constant followed by transpose(-1, -2)
    This is commonly found in transformer attention computations for key tensors.
    """
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for fused division + transpose operation
@triton.jit
def fused_div_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs division by constant and transpose in one operation.
    Transposes from [batch, heads, seq, head_dim] to [batch, heads, head_dim, seq]
    """
    pid = tl.program_id(0)
    
    # Calculate total elements per batch-head combination
    elements_per_bh = seq_len * head_dim
    bh_elements = pid * elements_per_bh
    
    # Calculate offsets within this batch-head group
    local_idx = tl.arange(0, BLOCK_SIZE)
    global_idx = bh_elements + local_idx
    
    # Compute indices for input and output
    # Input: [batch, heads, seq, head_dim]
    # Output: [batch, heads, head_dim, seq]
    
    # Calculate coordinates
    total_elements = batch_size * num_heads * seq_len * head_dim
    mask = global_idx < total_elements
    
    if not mask[0]:
        return  # Early exit for out-of-bounds threads
    
    # Convert linear index to 4D coordinates for input tensor
    # input shape: [batch_size, num_heads, seq_len, head_dim]
    linear_idx = global_idx
    offset_in_bh = linear_idx % elements_per_bh
    head_idx = (linear_idx // elements_per_bh) % num_heads
    batch_idx = linear_idx // (num_heads * elements_per_bh)
    
    seq_idx = offset_in_bh // head_dim
    head_dim_idx = offset_in_bh % head_dim
    
    # For output shape [batch_size, num_heads, head_dim, seq]
    # We need to swap seq_idx and head_dim_idx
    output_seq_idx = head_dim_idx
    output_head_dim_idx = seq_idx
    
    # Calculate output linear index
    output_offset_in_bh = output_head_dim_idx * seq_len + output_seq_idx
    output_linear_idx = batch_idx * num_heads * elements_per_bh + head_idx * elements_per_bh + output_offset_in_bh
    
    # Load input, scale, and store to output location with transpose
    input_val = tl.load(input_ptr + linear_idx, mask=mask, other=0.0)
    scaled_val = input_val * scale
    tl.store(output_ptr + output_linear_idx, scaled_val, mask=mask)

@torch.fx.wrap
def fused_div_transpose_gpu(input_tensor, scale=1.0/1.6817928305074292):
    """
    GPU-optimized fused operation that combines division and transpose.
    Transposes from [batch, heads, seq, head_dim] to [batch, heads, head_dim, seq]
    """
    shape = input_tensor.shape
    batch_size, num_heads, seq_len, head_dim = shape
    
    total_elements = input_tensor.numel()
    
    # Choose optimal block size based on tensor dimensions
    if head_dim >= 64:
        BLOCK_SIZE = 256
    elif head_dim >= 32:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 64
    
    # Calculate grid size
    elements_per_bh = seq_len * head_dim
    num_bh_groups = (batch_size * num_heads * elements_per_bh + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with correct shape after transpose
    output_shape = (batch_size, num_heads, head_dim, seq_len)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_div_transpose_kernel[(num_bh_groups,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Optimized fallback for CPU or smaller tensors
def fused_div_transpose_cpu(input_tensor, scale=1.0/1.6817928305074292):
    """
    CPU fallback using regular operations
    """
    scaled = input_tensor * scale
    result = scaled.transpose(-1, -2)
    return result

def replacement_func():
    """
    Returns the optimized function that will replace the original pattern
    """
    # Choose implementation based on device
    def optimized_replacement(in_0):
        if in_0.device.type == 'cuda':
            return fused_div_transpose_gpu(in_0)
        else:
            return fused_div_transpose_cpu(in_0)
    
    return optimized_replacement