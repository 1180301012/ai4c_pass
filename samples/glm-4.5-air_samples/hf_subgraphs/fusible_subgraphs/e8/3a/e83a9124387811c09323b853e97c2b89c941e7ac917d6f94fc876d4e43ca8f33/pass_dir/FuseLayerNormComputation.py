import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # LayerNorm computation pattern: ((in_3 + in_2) * in_1) + in_0
    # Return the fused result only
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    tensor1_ptr, 
    tensor2_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the hidden dimension
    hidden_idx = tl.program_id(0)
    batch_seq_idx = tl.program_id(1)
    
    # Calculate global memory offset
    offset = batch_seq_idx * hidden_size + hidden_idx
    
    # Load bias and weight (broadcast across batch and sequence)
    bias = tl.load(bias_ptr + hidden_idx, mask=hidden_idx < hidden_size, other=0.0)
    weight = tl.load(weight_ptr + hidden_idx, mask=hidden_idx < hidden_size, other=0.0)
    
    # Load tensor elements
    batch_seq_total = batch_size * seq_len
    valid_batch_seq = batch_seq_idx < batch_seq_total
    valid_hidden = hidden_idx < hidden_size
    valid_mask = valid_batch_seq & valid_hidden
    
    tensor1_val = tl.load(tensor1_ptr + offset, mask=valid_mask, other=0.0)
    tensor2_val = tl.load(tensor2_ptr + offset, mask=valid_mask, other=0.0)
    
    # Fused computation: ((tensor2 + tensor1) * weight) + bias
    result = ((tensor2_val + tensor1_val) * weight) + bias
    
    # Store result
    tl.store(out_ptr + offset, result, mask=valid_mask)

def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    return 2 ** (n - 1).bit_length()

@torch.fx.wrap  
def fused_layernorm_computation(in_0, in_1, in_2, in_3):
    # Determine tensor shapes
    if len(in_0.shape) == 1:  # [hidden_size]
        hidden_size = in_0.shape[0]
        # Calculate batch_size and seq_len from in_2/in_3 shapes
        batch_size = in_2.shape[0]
        seq_len = in_2.shape[1] if len(in_2.shape) > 2 else 1
    else:
        # Handle other cases if needed
        hidden_size = in_0.shape[-1]
        batch_size = in_2.numel() // hidden_size
        seq_len = 1
    
    # Prepare output tensor with same shape as input tensors
    out = torch.empty_like(in_2)
    
    # Use Triton's auto-tuning to find optimal block size
    # For hidden dimension, typical block sizes range from 32 to 1024
    # We'll use a moderately large block size for good occupancy
    BLOCK_SIZE = 128
    
    # Calculate grid size - each program handles one hidden element
    grid = (hidden_size, batch_size * seq_len)
    
    # Launch kernel with automatic grid optimization
    fused_layernorm_kernel[grid](
        bias_ptr=in_0,
        weight_ptr=in_1,
        tensor1_ptr=in_2,
        tensor2_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_layernorm_computation