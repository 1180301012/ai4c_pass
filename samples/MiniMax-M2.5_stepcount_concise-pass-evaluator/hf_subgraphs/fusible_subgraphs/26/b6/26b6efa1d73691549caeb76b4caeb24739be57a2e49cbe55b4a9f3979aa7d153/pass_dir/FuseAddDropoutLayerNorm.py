import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_dropout_ln_kernel(
    # Input pointers
    in_2_ptr,  # embeddings
    in_3_ptr,  # position_embeddings
    in_1_ptr,  # layer_norm weight
    in_0_ptr,  # layer_norm bias
    # Output pointers
    dropout_out_ptr,  # tmp_3 output
    ln_out_ptr,       # tmp_4 output
    # Dimensions
    batch_size,
    seq_len,
    hidden_dim,
    # Layer norm parameters
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. in_2 + in_3 (element-wise add)
    2. dropout (training=False, so identity)
    3. layer_norm
    """
    # Get position in the tensor
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Calculate offsets
    row_offset = row_idx * seq_len
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks
    col_mask = col_offsets < hidden_dim
    
    # Load in_2 (embeddings) and in_3 (position embeddings)
    in_2 = tl.load(in_2_ptr + row_offset * hidden_dim + col_offsets, mask=col_mask, other=0.0)
    in_3 = tl.load(in_3_ptr + row_offset * hidden_dim + col_offsets, mask=col_mask, other=0.0)
    
    # Element-wise add: in_2 + in_3
    added = in_2 + in_3
    
    # Store dropout output (identity since training=False)
    tl.store(dropout_out_ptr + row_offset * hidden_dim + col_offsets, added, mask=col_mask)
    
    # Load layer norm weight and bias
    ln_weight = tl.load(in_1_ptr + col_offsets, mask=col_mask, other=0.0)
    ln_bias = tl.load(in_0_ptr + col_offsets, mask=col_mask, other=0.0)
    
    # Compute mean and variance for layer norm
    # Mean
    mean = tl.sum(added, axis=0) / hidden_dim
    # Variance: E[(x - mean)^2]
    diff = added - mean
    variance = tl.sum(diff * diff, axis=0) / hidden_dim
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize
    normalized = diff * inv_std
    # Scale and shift
    ln_out = normalized * ln_weight + ln_bias
    
    # Store layer norm output
    tl.store(ln_out_ptr + row_offset * hidden_dim + col_offsets, ln_out, mask=col_mask)


@torch.fx.wrap
def fused_add_dropout_ln(in_2, in_3, in_1, in_0, normalized_shape, eps):
    """
    Wrapper function for the fused kernel.
    
    Args:
        in_2: embeddings tensor [batch, seq_len, hidden_dim]
        in_3: position_embeddings tensor [batch or 1, seq_len, hidden_dim]
        in_1: layer_norm weight [hidden_dim]
        in_0: layer_norm bias [hidden_dim]
        normalized_shape: tuple (hidden_dim,)
        eps: float for layer norm
    
    Returns:
        tuple: (dropout_output, layer_norm_output)
    """
    batch_size, seq_len, hidden_dim = in_2.shape
    
    # Expand in_3 if needed (broadcast from [1, seq_len, hidden_dim] to [batch, seq_len, hidden_dim])
    if in_3.shape[0] == 1:
        in_3 = in_3.expand(batch_size, -1, -1)
    
    # Allocate output tensors
    dropout_out = torch.empty_like(in_2)
    ln_out = torch.empty_like(in_2)
    
    # Configure kernel
    BLOCK_SIZE = 1024
    # Grid: (batch * seq_len, num_blocks)
    num_rows = batch_size * seq_len
    num_col_blocks = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_rows, num_col_blocks)
    
    fused_add_dropout_ln_kernel[grid](
        in_2, in_3, in_1, in_0,
        dropout_out, ln_out,
        batch_size, seq_len, hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dropout_out, ln_out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: add + dropout + layer_norm
    
    The computation is:
    1. in_2 += in_3 (element-wise add, in-place)
    2. tmp_3 = dropout(in_2, p=0.1, training=False, inplace=False)
    3. tmp_4 = layer_norm(tmp_3, normalized_shape, weight=in_1, bias=in_0, eps=1e-12)
    
    Returns:
        tuple: (dropout_output, layer_norm_output)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    in_2 += in_3
    tmp_2 = in_2
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    tmp_2 = None
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (64,), tmp_1, tmp_0, 1e-12)
    tmp_1 = tmp_0 = None
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    # Extract arguments needed for replacement
    # normalized_shape is (64,) for tiny models, (1024,) for large models
    # But we can get it from in_1.shape
    normalized_shape = (in_1.shape[0],)
    eps = 1e-12
    return (in_2, in_3, in_1, in_0, normalized_shape, eps)


def replacement_func():
    return fused_add_dropout_ln