import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr,                  # Input tensor [heads, seq_len, hidden]
    cos_sin_ptr,            # Cosine/sine embeddings [seq_len, hidden//2]
    qk_split_ptr,           # Split for q/k [hidden//2] (second branch only)
    out_ptr,                # Output tensor [heads, seq_len, hidden]
    n_heads,                # Number of heads
    n_seq,                  # Sequence length  
    n_hidden,               # Hidden dimension
    is_second_branch: tl.constexpr,  # Whether this is the second branch
    BLOCK_SIZE_M: tl.constexpr,      # Block size for M dimension (sequence)
    BLOCK_SIZE_N: tl.constexpr,      # Block size for N dimension (hidden)
    BLOCK_SIZE_K: tl.constexpr,      # Block size for K dimension (heads)
):
    # Calculate program IDs
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1) 
    k_pid = tl.program_id(2)
    
    # Calculate memory offsets
    m_offsets = m_pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = k_pid * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) if n_hidden > 1 else tl.arange(0, 1)
    
    # Create masks
    m_mask = m_offsets < n_seq
    n_mask = n_offsets < n_hidden
    k_mask = k_offsets < n_heads
    
    # Load input data - reshape to [heads, seq_len, hidden] for easier access
    x = tl.load(x_ptr + (k_offsets[:, None, None] * n_seq * n_hidden + 
                        m_offsets[None, :, None] * n_hidden + 
                        n_offsets[None, None, :]), 
                mask=k_mask[:, None, None] & m_mask[None, :, None] & n_mask[None, None, :], 
                other=0.0)
    
    # Load cosine/sine embeddings - reshape to [seq_len, hidden//2] for pairs
    cos_sin = tl.load(cos_sin_ptr + (m_offsets[:, None] * (n_hidden // 2) + 
                                     n_offsets[:, :n_hidden//2][None, :]), 
                     mask=m_mask[:, None] & (n_offsets[:n_hidden//2][None, :] < (n_hidden // 2)), 
                     other=0.0)
    
    # For second branch, also load the positional embedding split
    if is_second_branch:
        qk_split = tl.load(qk_split_ptr + n_offsets[:n_hidden//2], 
                          mask=n_offsets[:n_hidden//2] < (n_hidden // 2), 
                          other=0.0)
        qk_split = qk_split[None, None, :]  # Add batch and head dimensions
    
    # Apply RoPE operation: Extract even/odd indices, apply rotation, combine
    # This is the core RoPE logic: x_rotated = [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
    
    # Extract even and odd indices from hidden dimension
    if n_hidden >= 2:
        x_even = x[..., :n_hidden//2]  # Even indices
        x_odd = x[..., n_hidden//2:]   # Odd indices
        
        # Apply rotation using cosine and sine
        if is_second_branch:
            # Second branch use qk_split as the multiplier
            cos_term = cos_sin[..., 0::2]  # Every other row starting from 0 (cos)
            sin_term = cos_sin[..., 1::2]  # Every other row starting from 1 (sin)
            
            # Rotate odd indices with split
            rotated_even = x_even * cos_term - x_odd * sin_term * qk_split
            rotated_odd = x_even * sin_term * qk_split + x_odd * cos_term
        else:
            # First branch use standard cos/sin directly
            cos_term = cos_sin
            sin_term = tl.shift_right(cos_sin, 1)  # Get sin terms (shifted by 1)
            
            # Standard RoPE rotation
            rotated_even = x_even * cos_term - x_odd * sin_term
            rotated_odd = x_even * sin_term + x_odd * cos_term
        
        # Combine rotated even and odd indices
        out_rotated = tl.zeros([n_heads, n_seq, n_hidden], dtype=tl.float32)
        out_rotated = tl.index_put(out_rotated, (slice(None), slice(None), slice(0, n_hidden//2)), rotated_even)
        out_rotated = tl.index_put(out_rotated, (slice(None), slice(None), slice(n_hidden//2, n_hidden)), rotated_odd)
    else:
        # Handle case with hidden dimension 1 (fallback)
        out_rotated = x
    
    # Store results
    tl.store(out_ptr + (k_offsets[:, None, None] * n_seq * n_hidden + 
                        m_offsets[None, :, None] * n_hidden + 
                        n_offsets[None, None, :]), 
             out_rotated,
             mask=k_mask[:, None, None] & m_mask[None, :, None] & n_mask[None, None, :])

# Simple pattern to test basic functionality
def pattern(x, y):
    """Simple addition pattern for testing"""
    return x + y

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Simple triton addition for testing
@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute and store
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_triton_add(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y, 
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function 
def replacement_func():
    return simple_triton_add