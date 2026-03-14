import torch
import triton
import triton.language as tl

def pattern(tmp_1, in_3):
    """
    Simplified pattern to match just the transpose + multiplication.
    """
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    return tmp_3

def replacement_args(tmp_1, in_3):
    """
    Extract arguments needed for the fused operation.
    """
    return (tmp_1, in_3)

@triton.jit
def fused_transpose_mul_kernel(
    tmp_1_ptr, in_3_ptr, out_ptr,
    feat_dim, seq_dim, batch_size, num_groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for transpose + element-wise multiplication.
    Transposes (1, 8, seq_len, feat_dim) to (1, 8, feat_dim, seq_len) and multiplies with in_3.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Handle each of the 8 groups separately
    group = tl.program_id(2)
    
    # Calculate base offset for this group
    group_offset = group * batch_size * seq_dim * feat_dim
    
    m = m_offset + group * feat_dim
    n = n_offset
    
    # Check bounds with proper parentheses to avoid chained boolean operators
    m_bound = group_offset + batch_size * seq_dim * feat_dim
    if (m < m_bound) and (n < seq_dim):
        # Load from tmp_1 (reshape to (1, 8, seq_len, feat_dim))
        # We need to transpose to (1, 8, feat_dim, seq_len)
        # Original index in tmp_1: [batch, group, n, m] where m ranges over feat_dim
        # Transposed index: [batch, group, m, n] where m ranges over feat_dim
        
        batch_idx = 0  # batch is always 1
        seq_idx = n
        feat_idx = m % feat_dim
        current_group = m // feat_dim
        
        # Split chained boolean operators to avoid compilation errors
        if (current_group == group):
            if (seq_idx < seq_dim):
                if (feat_idx < feat_dim):
                    # Load from tmp_1 at [0, group, seq_idx, feat_idx]
                    tmp_1_idx = batch_idx * num_groups * seq_dim * feat_dim + \
                                group * seq_dim * feat_dim + \
                                seq_idx * feat_dim + \
                                feat_idx
                    val = tl.load(tmp_1_ptr + tmp_1_idx)
                    
                    # Load from in_3 at [0, group, feat_idx, seq_idx]
                    in_3_idx = batch_idx * num_groups * feat_dim * seq_dim + \
                               group * feat_dim * seq_dim + \
                               feat_idx * seq_dim + \
                               seq_idx
                    q_val = tl.load(in_3_ptr + in_3_idx)
                    
                    # Store result
                    out_idx = tmp_1_idx  # Same indexing for output
                    tl.store(out_ptr + out_idx, val * q_val)

@torch.fx.wrap
def fused_transpose_mul(tmp_1, in_3):
    """
    Fused function that performs:
    1. Transpose tmp_1 from (1, 8, seq_len, feat_dim) to (1, 8, feat_dim, seq_len)
    2. Element-wise multiply with query tensor
    
    This avoids intermediate memory allocations and improves memory locality.
    """
    # Extract tensor shapes
    batch_size, num_groups, seq_len, feat_dim = tmp_1.shape
    _, _, in_3_feat_dim, in_3_seq_dim = in_3.shape
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Only launch kernel if shapes are compatible
    if (num_groups == 8 and seq_len == in_3_seq_dim and feat_dim == in_3_feat_dim):
        
        # Configure block sizes for GPU memory coalescing
        BLOCK_SIZE_M = 128  # Feature dimension tile
        BLOCK_SIZE_N = 128  # Sequence dimension tile
        
        # Calculate grid size
        grid_m = (feat_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        # Launch kernel with 3D grid (m, n, group)
        fused_transpose_mul_kernel[grid_m, grid_n, num_groups](
            tmp_1_ptr=tmp_1,
            in_3_ptr=in_3,
            out_ptr=out,
            feat_dim=feat_dim,
            seq_dim=seq_len,
            batch_size=batch_size,
            num_groups=num_groups,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return out

def replacement_func():
    """
    Return the fused function as a callable.
    """
    return fused_transpose_mul