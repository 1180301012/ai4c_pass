import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern from model.py
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return (tmp_6,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def weighted_sum_norm_kernel(
    in_0_ptr,  # int64 tensor converted to float32 access
    in_1_ptr,  # bfloat16/float16 tensor
    out_ptr,   # output tensor
    n_batch,   # batch size
    n_seq,     # sequence length (10) - this is the dimension being summed over
    n_hidden,  # hidden dimension (1024) - this is the dimension being preserved
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Each program handles one element of the output (batch, hidden)
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)
    
    # Initialize accumulators
    weighted_sum = 0.0
    weight_sum = 0.0
    
    # Iterate over sequence dimension (10) in blocks
    for offset_seq in tl.range(0, n_seq, BLOCK_SIZE_SEQ):
        # Load in_0 values for this (batch, offset_seq, hidden)
        in_0_seq_idx = offset_seq + tl.arange(0, BLOCK_SIZE_SEQ)
        mask_in_0 = in_0_seq_idx < n_seq
        in_0_indices = batch_idx * n_seq * n_hidden + in_0_seq_idx * n_hidden + hidden_idx
        in_0_vals = tl.load(in_0_ptr + in_0_indices, 
                           mask=mask_in_0, 
                           other=0.0).to(tl.float32)
        
        # Load in_1 values for this (batch, offset_seq, hidden)
        in_1_indices = batch_idx * n_seq * n_hidden + in_0_seq_idx * n_hidden + hidden_idx
        in_1_vals = tl.load(in_1_ptr + in_1_indices, 
                           mask=mask_in_0, 
                           other=0.0).to(tl.float32)
        
        # Compute weighted sum and weight sum for this block and accumulate
        weighted_sum += tl.sum(in_1_vals * in_0_vals)
        weight_sum += tl.sum(in_0_vals)
    
    # Apply normalization with clamping  
    clamped_weight_sum = tl.maximum(weight_sum, 1e-09)
    result = weighted_sum / clamped_weight_sum
    
    # Store the result at the correct output location
    output_index = batch_idx * n_hidden + hidden_idx
    tl.store(out_ptr + output_index, result)

@torch.fx.wrap
def fused_weighted_sum_norm(in_0, in_1):
    # Get tensor shapes
    n_batch, n_seq, n_hidden = in_0.shape
    
    # Create output tensor - if error says expected is 1024, maybe I need that shape
    # Let me try producing [1, 1024] to see if that matches
    out = torch.empty((n_batch, n_hidden), dtype=torch.float32, device=in_0.device)
    
    # Choose block size for good GPU occupancy (for sequence dimension) - must be power of 2
    BLOCK_SIZE_SEQ = 8  # Use 8 (power of 2) - optimal for sequence length 10
    
    # Launch kernel - now programs handle (batch, hidden) pairs instead of (batch, seq)
    grid = (n_batch, n_hidden)
    weighted_sum_norm_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_hidden=n_hidden,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
    
    return out

def replacement_func():
    return fused_weighted_sum_norm