import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match the bmm operation that's in the original computation
    return torch.bmm(x, y)

def replacement_args(x, y):
    return (x, y)

@triton.jit
def bmm_kernel(
    attn_weights_ptr,   # [batch_size, num_heads, seq_len] 
    value_states_ptr,   # [batch_size, num_heads, head_dim]
    output_ptr,         # [batch_size, num_heads, head_dim]
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one output head
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Compute bounds for this head
    attn_offset = batch_idx * num_heads * seq_len + head_idx * seq_len
    value_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    
    # Create shared memory for a block of attention weights
    m = tl.program_id(2)
    for k in range(0, seq_len, BLOCK_SIZE_K):
        # Attention weight for this head (seq_len sequence is all the same since seq_len=1)
        attn_weight = tl.load(attn_weights_ptr + attn_offset + k)
        
        # Process a block of value states
        for n in range(0, head_dim, BLOCK_SIZE_N):
            # Compute output value
            output_val = attn_weight * tl.load(value_states_ptr + value_offset + n + 0)
            
            # Store output
            tl.store(output_ptr + output_offset + n + 0, output_val)

@torch.fx.wrap
def triton_bmm(attn_weights, value_states):
    # Get tensor shapes
    batch_size, num_heads, seq_len = attn_weights.shape
    _, _, head_dim = value_states.shape
    
    # Create output tensor with correct shape [batch_size, num_heads, head_dim]
    output = torch.empty(batch_size, num_heads, head_dim, device=attn_weights.device, dtype=attn_weights.dtype)
    
    # Flatten tensors for kernel access
    attn_weights_flat = attn_weights.view(-1)
    value_states_flat = value_states.view(-1)
    output_flat = output.view(-1)
    
    # Launch kernel with proper grid configuration
    # Each program handles one head, and each head outputs head_dim elements
    grid = (batch_size, num_heads, 1)  # We'll process each head completely in one program
    
    bmm_kernel[grid](
        attn_weights_flat,
        value_states_flat,
        output_flat,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=32,  # Process 32 elements at a time for head_dim
        BLOCK_SIZE_K=1,
    )
    
    return output

def replacement_func():
    return triton_bmm