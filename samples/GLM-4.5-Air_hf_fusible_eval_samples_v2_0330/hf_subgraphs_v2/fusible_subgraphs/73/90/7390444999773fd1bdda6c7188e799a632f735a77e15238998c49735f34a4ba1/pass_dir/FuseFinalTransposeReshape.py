import torch
import triton
import triton.language as tl

def pattern(in_3):
    """
    Pattern matches: attention_output.transpose(1, 2).reshape(final_shape)
    Returns the final reshaped output
    """
    tmp_6 = in_3.transpose(1, 2)  # in_3 is the attention output
    tmp_7 = tmp_6.reshape(1, 512, 128)
    return tmp_7

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def fused_transpose_reshape_kernel(
    attn_output_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    attn_seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    final_seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that combines transpose and reshape operations"""
    pid = tl.program_id(0)
    
    # Calculate output position
    output_pos = pid * BLOCK_SIZE
    output_end = min(output_pos + BLOCK_SIZE, batch_size * final_seq_len * hidden_dim)
    
    if output_pos >= batch_size * final_seq_len * hidden_dim:
        return
    
    # Process output elements
    for i in range(output_pos, output_end):
        # Calculate output coordinates
        batch_idx = i // (final_seq_len * hidden_dim)
        seq_idx = (i % (final_seq_len * hidden_dim)) // hidden_dim
        hidden_idx = i % hidden_dim
        
        # Map to input coordinates: [batch, attn_seq_len, hidden] -> [batch, num_heads, seq//num_heads, head_dim] -> [batch, seq//num_heads, num_heads, head_dim]
        # Here: batch=1, attn_seq_len=256, num_heads=2, head_dim=64, final_seq_len=512, hidden_dim=128
        
        # Input is [batch, num_heads, attn_seq_len, head_dim] after transpose
        # We want [batch, final_seq_len, hidden_dim]
        # final_seq_len = 512, attn_seq_len = 256, hidden_dim = 128
        # So we need to expand the attention output
        
        # Calculate positions in attention output
        attn_seq_pos = seq_idx % attn_seq_len  # Position within attention sequence
        attn_seq_expansion = seq_idx // attn_seq_len  # Which expansion (0 or 1)
        
        # Map to attention output index
        attn_idx = (batch_idx * num_heads * attn_seq_len + attn_seq_expansion * attn_seq_len + attn_seq_pos) * head_dim + hidden_idx % head_dim
        
        # Load from attention output and store to final output
        if attn_idx < batch_size * num_heads * attn_seq_len * head_dim:
            value = tl.load(attn_output_ptr + attn_idx)
            tl.store(output_ptr + i, value)

@triton.jit
def optimized_transpose_reshape_kernel(
    attn_output_ptr,  # [batch, num_heads, attn_seq_len, head_dim]
    output_ptr,      # [batch, final_seq_len, hidden_dim]
    batch_size: tl.constexpr,
    attn_seq_len: tl.constexpr,    # e.g., 256
    num_heads: tl.constexpr,       # e.g., 2
    head_dim: tl.constexpr,        # e.g., 64
    final_seq_len: tl.constexpr,   # e.g., 512
    hidden_dim: tl.constexpr,      # e.g., 128
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized kernel that combines transpose and reshape efficiently"""
    pid = tl.program_id(0)
    
    # Calculate dimensions for this tile
    batch_idx = pid // (final_seq_len * hidden_dim // (BLOCK_SIZE_M * BLOCK_SIZE_N))
    local_pid = pid % (final_seq_len * hidden_dim // (BLOCK_SIZE_M * BLOCK_SIZE_N))
    
    m = local_pid // BLOCK_SIZE_N
    n = local_pid % BLOCK_SIZE_N
    
    start_m = m * BLOCK_SIZE_M
    start_n = n * BLOCK_SIZE_N
    
    end_m = min(start_m + BLOCK_SIZE_M, final_seq_len)
    end_n = min(start_n + BLOCK_SIZE_N, hidden_dim)
    
    # Process each tile
    for i in range(start_m, end_m):
        for j in range(start_n, end_n):
            # Map output position [batch, seq, hidden] to attention position [batch, num_heads, attn_seq, head_dim]
            # Output: batch=1, seq=512, hidden=128
            # Attention: batch=1, num_heads=2, attn_seq=256, head_dim=64
            
            # We need to duplicate each attention output element across multiple positions
            # since final_seq_len (512) > attn_seq_len (256)
            
            # Calculate which attention sequence position this maps to
            attn_seq_pos = i % attn_seq_len
            expansion_factor = i // attn_seq_len  # 0 or 1 since 512 = 2 * 256
            
            if expansion_factor < num_heads:
                # Map to attention output index
                attn_idx = (batch_idx * num_heads * attn_seq_len + expansion_factor * attn_seq_len + attn_seq_pos) * head_dim + j
                
                if attn_idx < batch_size * num_heads * attn_seq_len * head_dim:
                    value = tl.load(attn_output_ptr + attn_idx)
                    output_idx = batch_idx * final_seq_len * hidden_dim + i * hidden_dim + j
                    tl.store(output_ptr + output_idx, value)

@torch.fx.wrap
def fused_transpose_reshape(attn_output):
    """Wrapper function that transposes and reshapes attention output"""
    batch_size, num_heads, attn_seq_len, head_dim = attn_output.shape
    
    # Determine output shape from the model pattern
    # From the model: reshape(1, 512, 128)
    final_seq_len = 512
    hidden_dim = 128
    
    assert batch_size == 1, f"Only batch size 1 supported, got {batch_size}"
    assert num_heads * attn_seq_len == final_seq_len, f"Size mismatch: {num_heads} * {attn_seq_len} != {final_seq_len}"
    assert head_dim == hidden_dim, f"Head dimension mismatch: {head_dim} != {hidden_dim}"
    
    output_shape = (batch_size, final_seq_len, hidden_dim)
    output = torch.empty(output_shape, dtype=attn_output.dtype, device=attn_output.device)
    
    # Launch kernel
    block_size = 256
    total_elements = batch_size * final_seq_len * hidden_dim
    num_programs = (total_elements + block_size - 1) // block_size
    
    fused_transpose_reshape_kernel[(num_programs,)](
        attn_output,
        output,
        batch_size,
        attn_seq_len,
        num_heads,
        head_dim,
        final_seq_len,
        hidden_dim,
        block_size,
    )
    
    return output

@torch.fx.wrap  
def optimized_transpose_reshape(attn_output):
    """More optimized version with better memory access patterns"""
    batch_size, num_heads, attn_seq_len, head_dim = attn_output.shape
    
    # Determine output shape from the model pattern
    final_seq_len = 512
    hidden_dim = 128
    
    assert batch_size == 1, f"Only batch size 1 supported, got {batch_size}"
    assert num_heads * attn_seq_len == final_seq_len, f"Size mismatch: {num_heads} * {attn_seq_len} != {final_seq_len}"
    assert head_dim == hidden_dim, f"Head dimension mismatch: {head_dim} != {hidden_dim}"
    
    output_shape = (batch_size, final_seq_len, hidden_dim)
    output = torch.empty(output_shape, dtype=attn_output.dtype, device=attn_output.device)
    
    # Launch kernel with tiling
    BLOCK_SIZE_M = 32  # sequence dimension
    BLOCK_SIZE_N = 32  # hidden dimension
    
    grid_m = tl.cdiv(final_seq_len, BLOCK_SIZE_M)
    grid_n = tl.cdiv(hidden_dim, BLOCK_SIZE_N)
    total_programs = grid_m * grid_n
    
    optimized_transpose_reshape_kernel[(total_programs,)](
        attn_output,
        output,
        batch_size,
        attn_seq_len,
        num_heads,
        head_dim,
        final_seq_len,
        hidden_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_transpose_reshape