import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """ 
    Matches: scalar multiply + broadcasting addition + softmax fusion
    This pattern captures the core computation: in_1 * scale + in_0.unsqueeze(2) + softmax
    """
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    tmp_3 = tmp_2.softmax(dim=-1)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_1, in_0, 0.1767766952966369)

# Optimized kernel using Triton
@triton.jit
def fused_kernel(
    in_1_ptr,
    in_0_ptr,
    output_ptr,
    scale,
    batch_size,
    num_heads,
    k_dim, 
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * num_heads * k_dim * seq_len * head_dim
    
    # Calculate tensor dimensions for 5D tensor [batch, heads, k_dim, seq_len, head_dim]
    elem_idx = offsets
    batch_idx = elem_idx // (num_heads * k_dim * seq_len * head_dim)
    heads_idx = (elem_idx // (k_dim * seq_len * head_dim)) % num_heads
    k_idx = (elem_idx // (seq_len * head_dim)) % k_dim
    seq_idx = (elem_idx // head_dim) % seq_len
    head_idx = elem_idx % head_dim
    
    # Load input_1: [batch, heads, k_dim, seq_len, head_dim]
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Load input_0 and broadcast: input_0 is [batch, heads, seq_len, head_dim]
    # We need to broadcast it to [batch, heads, k_dim, seq_len, head_dim]
    # Calculate the original flattened index for input_0 (without k_dim)
    in_0_flat_idx = (batch_idx * num_heads + heads_idx) * seq_len * head_dim + seq_idx * head_dim + head_idx
    in_0_broadcast_val = tl.load(in_0_ptr + in_0_flat_idx, mask=mask, other=0.0)
    
    # Fused computation: scale multiply + broadcast add + softmax
    # Step 1: Scale multiplication
    scaled_in_1 = in_1_val * scale
    
    # Step 2: Broadcasting addition (input_0 gets broadcast across k_dim)
    fused_val = scaled_in_1 + in_0_broadcast_val
    
    # Step 3: Softmax along the last dimension (head_dim)
    # For each position in batch, heads, k_dim, seq_len, apply softmax across head_dim
    max_val = tl.max(fused_val, axis=4)
    exp_val = tl.exp(fused_val - max_val)
    sum_exp = tl.sum(exp_val, axis=4, keep_dims=True)
    softmax_out = exp_val / sum_exp
    
    # Store the result
    tl.store(output_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_scalar_mul_broadcast_add_softmax(in_1, in_0, scale):
    # Get tensor dimensions: in_1 is [batch, heads, k_dim, seq_len, head_dim]
    batch_size, num_heads, k_dim, seq_len, head_dim = in_1.shape
    total_elements = batch_size * num_heads * k_dim * seq_len * head_dim
    
    # Create output tensor
    output = torch.empty_like(in_1)
    
    # Set block size and grid size
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch the kernel
    fused_kernel[grid_size](
        in_1,
        in_0,
        output,
        scale,
        batch_size, num_heads, k_dim, seq_len, head_dim,
        BLOCK_SIZE
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_scalar_mul_broadcast_add_softmax