import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # This pattern matches the entire layer normalization computation chain
    tmp_1 = in_3 + in_2
    tmp_2 = in_1[-1]
    tmp_3 = tmp_2 + 1
    tmp_2 = tmp_3 = None
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_4 = None
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_5 = None
    tmp_7 = tmp_6 + 1e-06
    tmp_6 = None
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_7 = None
    tmp_9 = tmp_1 * tmp_8
    tmp_8 = None
    tmp_10 = in_0 * tmp_9
    tmp_0 = tmp_9 = None
    return (tmp_1, tmp_10)

def replacement_args(in_0, in_1, in_2, in_3):
    # Extract and return arguments needed for the replacement
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_layer_norm_kernel(
    dropout_ptr, hidden_states_ptr, weight_ptr,
    sum_out_ptr, norm_out_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Each program handles a block of the tensor
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute offsets for batch and sequence
    batch_idx = m
    seq_idx = n
    
    # Create masks
    mask_batch = batch_idx < batch_size
    mask_seq = seq_idx < seq_len
    
    # Load entire hidden dimension for this [batch, seq] location
    hidden_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size
    dropout_vals = tl.load(dropout_ptr + hidden_offset + tl.arange(0, hidden_size), 
                           mask=mask_batch & mask_seq, other=0.0)
    hidden_vals = tl.load(hidden_states_ptr + hidden_offset + tl.arange(0, hidden_size), 
                          mask=mask_batch & mask_seq, other=0.0)
    
    # Step 1: Perform addition (dropout + hidden_states)
    sum_vals = dropout_vals + hidden_vals
    
    # Step 2: Compute mean of squares over hidden dimension
    sum_squares = tl.sum(sum_vals * sum_vals)
    mean_squared = sum_squares / hidden_size
    
    # Add epsilon and compute rsqrt
    rsqrt_val = tl.rsqrt(mean_squared + 1e-06)
    
    # Step 3: Normalize and multiply by weight
    normalized_vals = sum_vals * rsqrt_val
    weighted_vals = normalized_vals * weight_ptr[tl.arange(0, hidden_size)]
    
    # Store results
    if mask_batch & mask_seq:
        # Store sum values
        tl.store(sum_out_ptr + hidden_offset + tl.arange(0, hidden_size), sum_vals, 
                mask=tl.arange(0, hidden_size) < hidden_size)
        
        # Store normalized and weighted values
        tl.store(norm_out_ptr + hidden_offset + tl.arange(0, hidden_size), weighted_vals, 
                mask=tl.arange(0, hidden_size) < hidden_size)

@torch.fx.wrap
def fused_layer_norm_forward(in_0, in_1, in_2, in_3):
    # Handle tensor shapes: in_2 and in_3 are [batch_size, seq_len, hidden_size]
    batch_size, seq_len, hidden_size = in_2.shape
    
    # Create output tensors
    sum_out = torch.empty_like(in_2)
    norm_out = torch.empty_like(in_2)
    
    # Launch Triton kernel - each program handles one [batch, seq] location
    fused_layer_norm_kernel[(batch_size, seq_len, 1)](
        in_2, in_3, in_0,
        sum_out, norm_out,
        batch_size, seq_len, hidden_size,
        1, 1  # Block sizes - we process one [batch, seq] location per kernel
    )
    
    return sum_out, norm_out

def replacement_func():
    return fused_layer_norm_forward