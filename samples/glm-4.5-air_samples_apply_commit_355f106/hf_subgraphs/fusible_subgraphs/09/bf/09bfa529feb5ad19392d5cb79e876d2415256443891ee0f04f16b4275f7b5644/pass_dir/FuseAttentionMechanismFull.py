import torch
import triton
import triton.language as tl
import math

@triton.jit
def row_softmax_kernel(
    x_ptr, out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    """Row-wise softmax kernel for attention mechanisms"""
    # Each program handles one element
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Calculate total elements
    total_elements = batch_size * num_heads * seq_len * head_dim
    mask = (row_idx < batch_size * num_heads * seq_len) & (col_idx < head_dim)
    element_idx = row_idx * head_dim + col_idx
    
    # Load the row (sequence position for attention head)
    x = tl.load(x_ptr + element_idx, mask=mask, other=-float('inf'))
    
    # Find max in the row (softmax along sequence dimension)
    max_val = x
    for i in range(1, seq_len):
        offset = row_idx * head_dim + i * head_dim + col_idx
        if offset < total_elements:
            val = tl.load(x_ptr + offset, mask=offset < total_elements, other=-float('inf'))
            max_val = tl.maximum(max_val, val)
    
    # Subtract max and exponentiate
    exp_x = tl.exp(x - max_val)
    
    # Store result
    tl.store(out_ptr + element_idx, exp_x, mask=mask)

@triton.jit
def fused_attention_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_heads, seq_len, head_dim,
    dropout_p, 
    BLOCK_SIZE: tl.constexpr
):
    """Fused attention kernel: addition + softmax + dropout + type conversion"""
    # Each program handles one element
    idx = tl.program_id(0)
    mask = idx < batch_size * num_heads * seq_len * head_dim
    
    # Load input elements
    in_0_val = tl.load(in_0_ptr + idx, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + idx, mask=mask, other=0.0)
    
    # Addition
    sum_val = in_0_val + in_1_val
    
    # Calculate row and column indices for softmax
    row_idx = idx // head_dim
    col_idx = idx % head_dim
    
    # Find max in the row (softmax along sequence dimension)
    max_val = sum_val
    for i in range(1, seq_len):
        offset = row_idx * head_dim + i * head_dim + col_idx
        if offset < batch_size * num_heads * seq_len * head_dim:
            val_0 = tl.load(in_0_ptr + offset, mask=offset < batch_size * num_heads * seq_len * head_dim, other=-float('inf'))
            val_1 = tl.load(in_1_ptr + offset, mask=offset < batch_size * num_heads * seq_len * head_dim, other=0.0)
            val_sum = val_0 + val_1
            max_val = tl.maximum(max_val, val_sum)
    
    # Subtract max and exponentiate
    exp_sum = tl.exp(sum_val - max_val)
    
    # Apply dropout - use scaling training=False behavior (no dropout during inference)
    if dropout_p > 0:
        # Create deterministic mask using element index
        # Note: This is a simplified dropout, real dropout might need proper random state
        seed = (idx * 9182 + 7331) & 0xFFFFFFFF  # Deterministic seed
        rand_val = tl.rand(seed)
        keep_prob = 1.0 - dropout_p
        keep_mask = rand_val > dropout_p
        exp_sum = exp_sum * keep_mask / keep_prob
    
    # Type conversion to float32
    result = tl.cast(exp_sum, tl.float32)
    
    # Store result
    tl.store(out_ptr + idx, result, mask=mask)

@torch.fx.wrap
def fused_attention_forward(in_0, in_1, dropout_p=0.1):
    """Forward pass with fused attention operations"""
    # Get tensor shapes - assuming [batch, num_heads, seq_len, head_dim] format
    batch_size, num_heads, seq_len, head_dim = in_0.shape
    n_elements = in_0.numel()
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Choose block size - based on typical GPU warp size
    BLOCK_SIZE = 1024
    
    # Calculate grid size - use 1D grid for simplicity
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_attention_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(a, b):
    """Pattern: addition -> softmax -> dropout -> type conversion"""
    tmp_0 = a + b
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_0 = None
    # Use positional arguments like the actual model
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    tmp_1 = None
    tmp_3 = tmp_2.to(torch.float32)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(a, b):
    """Extract arguments from matched nodes"""
    return (a, b)

def replacement_func():
    """Return the fused attention function"""
    return fused_attention_forward