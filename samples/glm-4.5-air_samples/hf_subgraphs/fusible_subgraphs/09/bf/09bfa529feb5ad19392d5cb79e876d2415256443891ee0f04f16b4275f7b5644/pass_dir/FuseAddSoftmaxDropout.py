import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    # Simple pattern: just match addition operation
    tmp_0 = in_0 + in_1
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_add_softmax_dropout_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len_q: tl.constexpr,
    seq_len_k: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data within a head
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len_k
    
    # Extract coordinates from flat index
    # Program ID determines which head we're processing
    program_id = tl.program_id(0)
    
    # Calculate head index and within-head offsets 
    # Note: This assumes a particular memory layout order
    head_idx = program_id % num_heads
    batch_idx = program_id // num_heads
    
    # Load tensors element by element for this head position
    stride_head = seq_len_q * seq_len_k
    stride_batch = num_heads * stride_head
    
    # Load input tensors for this head and batch position
    x_vals = []
    y_vals = []
    
    for i in tl.arange(0, seq_len_k):
        offset = (batch_idx * stride_batch + 
                 head_idx * stride_head + 
                 seq_len_q * i)
        x_vals.append(tl.load(x_ptr + offset))
        y_vals.append(tl.load(y_ptr + offset))
    
    # Convert to tensors and add
    x_tensor = tl.stack(x_vals)
    y_tensor = tl.stack(y_vals)
    z = x_tensor + y_tensor
    
    # Apply softmax along the seq_len_k dimension (last dimension)
    max_z = tl.max(z)
    shifted_z = z - max_z
    exp_z = tl.exp(shifted_z)
    sum_exp = tl.sum(exp_z)
    softmax_out = exp_z / sum_exp
    
    # Apply dropout if dropout_p > 0.0
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p > 0.0 else 1.0
    
    if dropout_p > 0.0:
        dropout_mask = tl.where(tl.random_uniform() > dropout_p, 1.0, 0.0)
        result = softmax_out * dropout_mask * dropout_scale
    else:
        result = softmax_out
    
    # Store result for each element in the sequence
    for i in tl.arange(0, seq_len_k):
        offset = (batch_idx * stride_batch + 
                 head_idx * stride_head + 
                 seq_len_q * i)
        if i < seq_len_k:  # Bounds check
            tl.store(out_ptr + offset, result[i])

@torch.fx.wrap
def fused_add_softmax_dropout(x, y, dropout_p=0.1):
    # Extract tensor dimensions based on the observed patterns
    # Shape is typically [batch_size, num_heads, seq_len_q, seq_len_k]
    batch_size = x.shape[0]
    num_heads = x.shape[1]
    seq_len_q = x.shape[2] 
    seq_len_k = x.shape[3]
    
    # Choose block size for within-head processing
    BLOCK_SIZE = min(256, seq_len_k)  # Process complete softmax rows
    
    # Calculate grid dimensions - one program per head
    total_heads = batch_size * num_heads
    num_programs = (total_heads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (float32 preserved from input)
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_add_softmax_dropout_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_softmax_dropout