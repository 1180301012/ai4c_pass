import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple pattern: just add two tensors
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    # Simple kernel that just adds two tensors
    @triton.jit
    def simple_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    
    @torch.fx.wrap
    def simple_add(x, y):
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty_like(x)
        simple_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
        return output
    
    return simple_add
def fused_value_projection_kernel(
    hidden_states_ptr,
    weight_ptr,
    bias_ptr,
    value_ptr,
    batch_size,
    seq_len,
    hidden_size,
    num_heads,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Each program handles one head in the batch
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Compute pointers for this batch and head
    batch_offset = batch_id * seq_len * hidden_size
    head_offset = head_id * head_dim
    
    # Create masks
    mask = (tl.arange(0, BLOCK_SIZE_M) < seq_len)[:, None]
    
    # Offset pointers
    h_ptr = hidden_states_ptr + batch_offset
    w_ptr = weight_ptr + head_id * head_dim * hidden_size
    v_ptr = value_ptr + batch_id * seq_len * num_heads * head_dim + head_id * seq_len * head_dim
    
    for k_base in range(0, hidden_size, BLOCK_SIZE_K):
        k_end = min(k_base + BLOCK_SIZE_K, hidden_size)
        
        # Load hidden states for this head
        h_offsets = tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_K)
        h_data = tl.load(h_ptr + h_offsets, mask=h_offsets < (seq_len * k_end), other=0.0)
        h_data = h_data.reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        
        # Load weight matrix for this head
        w_offsets = tl.arange(0, BLOCK_SIZE_K * BLOCK_SIZE_N)
        w_mask = (w_offsets < (k_end * BLOCK_SIZE_N))[:, None]
        w_data = tl.load(w_ptr + k_base * BLOCK_SIZE_N + w_offsets, mask=w_mask, other=0.0)
        w_data = w_data.reshape((BLOCK_SIZE_K, BLOCK_SIZE_N))
        
        # Matrix multiplication
        acc = tl.dot(h_data, w_data, out_dtype=tl.float32)
        
        # Add bias if provided
        if bias_ptr is not None:
            bias = tl.load(bias_ptr + head_offset + tl.arange(0, BLOCK_SIZE_N), mask=True, other=0.0)
            acc += bias
        
        # Store result
        v_offsets = tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N)
        v_mask = (v_offsets < (seq_len * BLOCK_SIZE_N))[:, None]
        tl.store(v_ptr + v_offsets, acc, mask=v_mask)

@torch.fx.wrap
def fused_value_projection(hidden_states, weight, bias, num_heads=12, head_dim=64):
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Determine head dimension based on hidden size and num_heads
    actual_head_dim = hidden_size // num_heads
    if actual_head_dim != head_dim:
        # Adjust head_dim parameters dynamically
        head_dim = actual_head_dim
    
    # Create output tensor
    output = torch.empty(batch_size, num_heads, seq_len, head_dim, 
                       device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 32  # Sequence length block
    BLOCK_SIZE_N = head_dim  # Head dimension block  
    BLOCK_SIZE_K = 64  # Hidden dimension block
    
    m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (batch_size, num_heads, m)
    
    # Launch kernel
    fused_value_projection_kernel[grid](
        hidden_states_ptr=hidden_states,
        weight_ptr=weight,
        bias_ptr=bias,
        value_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return fused_value_projection