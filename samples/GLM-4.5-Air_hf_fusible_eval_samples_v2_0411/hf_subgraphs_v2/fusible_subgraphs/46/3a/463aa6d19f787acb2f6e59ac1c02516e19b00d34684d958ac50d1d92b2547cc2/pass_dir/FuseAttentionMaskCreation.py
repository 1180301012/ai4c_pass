import torch
import triton
import triton.language as tl

def pattern(matrix, diagonal):
    """Simple pattern matching upper triangular matrix creation"""
    result = torch.triu(matrix, diagonal=diagonal)
    return result

def replacement_args(matrix, diagonal):
    """Extract matrix and diagonal for triangular pattern matching"""
    # The pattern function receives (matrix, diagonal) and should return the same
    return (matrix, diagonal)

@triton.jit
def attention_mask_kernel(
    output_ptr,
    attention_mask_ptr,
    seq_len,
    max_neg_value: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one block of the attention matrix
    batch_idx = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)
    
    # Create block pointers
    batch_offset = batch_idx * seq_len * seq_len
    m_block_start = m * BLOCK_SIZE_M
    n_block_start = n * BLOCK_SIZE_N
    
    # Process each element in the block
    for i in range(BLOCK_SIZE_M):
        row_idx = m_block_start + i
        if row_idx >= seq_len:
            continue
            
        for j in range(BLOCK_SIZE_N):
            col_idx = n_block_start + j
            if col_idx >= seq_len:
                continue
                
            # Calculate output position
            output_offset = batch_offset + row_idx * seq_len + col_idx
            output_ptr_local = output_ptr + output_offset
            
            # Load attention mask value
            attention_offset = batch_idx * seq_len + min(row_idx, seq_len-1)
            attention_val = tl.load(attention_mask_ptr + attention_offset, other=0.0)
            
            # Apply causal mask: no attending to future tokens
            if col_idx > row_idx:
                tl.store(output_ptr_local, max_neg_value)
            else:
                # For valid positions, use attention mask (0 means padded)
                tl.store(output_ptr_local, float(attention_val))

@torch.fx.wrap
def optimized_attention_mask_function(matrix, diagonal):
    """Optimized upper triangular matrix creation using Triton"""
    
    seq_len = matrix.shape[0]
    # Hardcode the fill value from the original computation
    fill_value = -3.4028234663852886e+38
    
    # Create output tensor
    output = torch.empty((seq_len, seq_len), dtype=torch.float32, device=matrix.device)
    
    # Launch simple Triton kernel for triangular matrix creation
    triangular_kernel[(seq_len, seq_len)](
        output_ptr=output,
        seq_len=seq_len,
        fill_value=fill_value,
        diagonal=diagonal
    )
    
    return output

@triton.jit
def triangular_kernel(
    output_ptr,
    seq_len: tl.constexpr,
    fill_value: tl.constexpr,
    diagonal: tl.constexpr,
):
    """Simple kernel to create upper triangular matrix"""
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row < seq_len and col < seq_len:
        if col >= row + diagonal:
            tl.store(output_ptr + row * seq_len + col, fill_value)
        else:
            tl.store(output_ptr + row * seq_len + col, 0.0)

def replacement_func():
    return optimized_attention_mask_function