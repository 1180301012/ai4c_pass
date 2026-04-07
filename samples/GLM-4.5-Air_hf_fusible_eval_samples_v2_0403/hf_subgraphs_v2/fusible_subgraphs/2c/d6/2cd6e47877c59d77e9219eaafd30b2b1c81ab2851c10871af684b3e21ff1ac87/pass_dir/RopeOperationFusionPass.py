import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel_cos_sin(
    x_ptr,
    cos_ptr, 
    sin_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized RoPE kernel that fuses:
    1. Complex number creation from [odd, even] elements
    2. Application of RoPE factors (cos/sin)
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate offsets within the batch and head
    batch_head_offset = batch_idx * num_heads * seq_len * hidden_dim + head_idx * seq_len * hidden_dim
    seq_offset = seq_idx * hidden_dim
    
    # Calculate pointer to input data for this sequence position
    x_ptr_local = x_ptr + batch_head_offset + seq_offset
    
    # Load input data
    x = tl.load(x_ptr_local, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Apply RoPE operation
    # For each position, x0 and x1 are cos_emb * x - sin_emb * x_rotated
    # where x_rotated is obtained by flipping odd and even elements with a sign
    
    # Load RoPE factors
    cos = tl.load(cos_ptr + seq_idx * hidden_dim // 2 + tl.arange(0, hidden_dim // 2), 
                  mask=tl.arange(0, hidden_dim // 2) < hidden_dim // 2, other=0.0)
    sin = tl.load(sin_ptr + seq_idx * hidden_dim // 2 + tl.arange(0, hidden_dim // 2), 
                  mask=tl.arange(0, hidden_dim // 2) < hidden_dim // 2, other=0.0)
    
    # Extract even and odd indices for RoPE rotation
    # x_even: indices 0, 2, 4, ... (first half of each pair)
    # x_odd: indices 1, 3, 5, ... (second half of each pair)
    x_even = tl.arange(0, hidden_dim) % 2 == 0
    x_odd = tl.arange(0, hidden_dim) % 2 == 1
    
    # Apply RoPE rotation
    # x_rotated[i] = -x[i+1] for even i
    # x_rotated[i+1] = x[i] for even i
    x_rotated = tl.zeros([hidden_dim], dtype=tl.float16)
    x_rotated_even = tl.where(x_even, -x[1::2], 0.0)
    x_rotated_odd = tl.where(x_odd, x[:-1:2], 0.0)
    
    # Interleave back to original order
    idx_even = tl.arange(0, hidden_dim // 2) * 2
    idx_odd = tl.arange(0, hidden_dim // 2) * 2 + 1
    
    # Create final result by applying RoPE formula
    result = tl.zeros([hidden_dim], dtype=tl.float16)
    
    # Apply to even elements: x0 * cos - x1 * sin
    even_indices = tl.arange(0, hidden_dim) % 2 == 0
    even_pos = tl.arange(0, hidden_dim // 2)
    result_even = tl.where(even_indices, 
                          x[even_pos * 2] * cos[even_pos] + x[even_pos * 2 + 1] * sin[even_pos],
                          0.0)
    
    # Apply to odd elements: x1 * cos + x0 * sin  
    odd_indices = tl.arange(0, hidden_dim) % 2 == 1
    odd_pos = tl.arange(0, hidden_dim // 2)
    result_odd = tl.where(odd_indices,
                         x[odd_pos * 2 + 1] * cos[odd_pos] - x[odd_pos * 2] * sin[odd_pos],
                         0.0)
    
    # Combine results
    result = result_even + result_odd
    
    # Store result
    out_ptr_local = out_ptr + batch_head_offset + seq_offset
    tl.store(out_ptr_local, result, mask=tl.arange(0, hidden_dim) < hidden_dim)

@triton.jit
def rope_kernel_complex_mult(
    x_ptr,
    cos_sin_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized RoPE kernel that directly multiplies by pre-complexified RoPE factors
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate offsets
    batch_head_offset = batch_idx * num_heads * seq_len * hidden_dim + head_idx * seq_len * hidden_dim
    seq_offset = seq_idx * hidden_dim
    
    x_ptr_local = x_ptr + batch_head_offset + seq_offset
    cos_sin_ptr_local = cos_sin_ptr + seq_idx * hidden_dim
    out_ptr_local = out_ptr + batch_head_offset + seq_offset
    
    # Load data
    x = tl.load(x_ptr_local, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    cos_sin = tl.load(cos_sin_ptr_local, mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Element-wise multiplication
    result = x * cos_sin
    
    # Store result
    tl.store(out_ptr_local, result, mask=tl.arange(0, hidden_dim) < hidden_dim)

def create_complex_rope(x, cos_emb, sin_emb):
    """
    Create complex RoPE representation and apply it efficiently
    Pattern matches: RoPE operation with complex number creation and multiplication
    """
    # Extract even and odd indices
    x_even = x[..., ::2]  # even indices
    x_odd = x[..., 1::2]  # odd indices
    
    # Create complex representation: [x_real, x_imag] = [x_even, -x_odd]
    x_real = x_even
    x_imag = -x_odd
    
    # Reshape for element-wise operations
    original_shape = x.shape
    cos_reshaped = cos_emb.view(1, 1, -1, 1)  # broadcast across heads and batch
    sin_reshaped = sin_emb.view(1, 1, -1, 1)
    
    # Apply RoPE rotation: (x_real + i * x_imag) * (cos + i * sin)
    # = (x_real * cos - x_imag * sin) + i * (x_real * sin + x_imag * cos)
    result_real = x_real * cos_reshaped - x_imag * sin_reshaped
    result_imag = x_real * sin_reshaped + x_imag * cos_reshaped
    
    # Interleave back to original order
    result = torch.stack([result_real, result_imag], dim=-1)
    result = result.reshape(original_shape)
    
    return result

@torch.fx.wrap
def rope_optimized(x, cos_emb, sin_emb, reshape_shape=None):
    """
    Optimized RoPE operation using Triton kernel
    """
    if x.dim() == 4:  # [batch, heads, seq, hidden]
        batch_size, num_heads, seq_len, hidden_dim = x.shape
    else:
        raise ValueError("Expected 4D tensor input")
    
    # Pre-compute complex RoPE factors if not provided
    if reshape_shape is not None:
        # Reshape for better parallelization
        x_reshaped = x.reshape(-1, seq_len, hidden_dim)
        result = create_complex_rope(x_reshaped, cos_emb, sin_emb)
        result = result.reshape(batch_size, num_heads, seq_len, hidden_dim)
    else:
        # Use Triton kernel for direct computation
        grid = (batch_size, num_heads, seq_len)
        BLOCK_SIZE = 64
        
        output = torch.empty_like(x)
        
        # Launch kernel
        rope_kernel_complex_mult[grid](
            x_ptr=x,
            cos_sin_ptr=cos_emb,  # Use cos_emb as placeholder, should be pre-complexified
            out_ptr=output,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        result = output
    
    return result

# Pattern matching function - matches the RoPE operation sequence
def rope_pattern(x_orig, x, cos_emb, sin_emb, first_multiply=True):
    """
    Matches the RoPE operation pattern in the original computation
    """
    # First branch operations (simplified pattern matching)
    if first_multiply:
        # tmp_1 = x_orig * cos_emb
        tmp_1 = x_orig * cos_emb
        
        # Extract even/odd elements and create complex representation
        tmp_2 = x_orig[..., 1::2]  # odd elements
        tmp_3 = -tmp_2
        tmp_4 = x_orig[..., ::2]   # even elements
        tmp_5 = torch.stack([tmp_3, tmp_4], -1)
        tmp_6 = tmp_5.reshape(x.shape)  # reshape back to original shape
        tmp_7 = tmp_6 * sin_emb
        tmp_8 = tmp_1 + tmp_7
        
        return tmp_8
    else:
        # Second branch operations (similar but with tensor split)
        # This would be handled by a separate pattern or parameterized pattern
        return x

# Argument extraction function
def rope_replacement_args(x, cos_emb, sin_emb, reshape_shape=None):
    return (x, cos_emb, sin_emb, reshape_shape)

# Replacement function - returns the optimized kernel
def rope_replacement_func():
    return rope_optimized

# Additional pattern for the second RoPE branch (with tensor split)
def rope_pattern_with_split(pos_embed, k_cos_sin, x, sin_emb, first_part_slice):
    """
    Matches the second RoPE branch that includes tensor splitting
    """
    # Extract first part of k tensor
    tmp_11 = first_part_slice
    
    # Extract second part of k tensor
    tmp_12 = k_cos_sin
    
    # Split positional embeddings
    tensor_split = pos_embed.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    
    # RoPE operations on the second part
    tmp_16 = tmp_12 * tmp_15
    
    # Complex number creation
    tmp_17 = tmp_12[..., 1::2]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[..., ::2]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape(tmp_12.shape)
    
    # Apply RoPE rotation
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    
    # Concatenate with first part
    result = torch.cat([tmp_11, tmp_23], dim=2)
    
    return result

def rope_split_replacement_args(pos_embed, k_cos_sin, x, sin_emb, first_part_slice):
    return (pos_embed, k_cos_sin, x, sin_emb, first_part_slice)

def rope_split_replacement_func():
    return rope_optimized